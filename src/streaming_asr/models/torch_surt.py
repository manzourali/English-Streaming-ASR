"""PyTorch composition for a trained, exported native streaming SURT model.

The factory loads its pinned recipe's weights into real DPRNN and Zipformer
modules. Both exports implement initial_state() and forward_chunk(x, state,
valid_frames, final), returning (output, state). Mask output is [1,W,160];
Zipformer output is [2,T,D] plus lengths: (output, lengths, state). Exports
must consume only the first 32 frames, keep bounded caches, use at most the
declared context, and emit all valid outputs on every call (no delayed tail).
Offline recipe modules do not satisfy this interface without a real export.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from torch import nn

from .multitalker import BackendOutput


def _state_bytes(state) -> int:
    if isinstance(state, torch.Tensor):
        if state.grad_fn is not None:
            raise ValueError("Streaming caches must be detached")
        return state.untyped_storage().nbytes()
    if isinstance(state, dict):
        return sys.getsizeof(state) + sum(sys.getsizeof(key) + _state_bytes(value) for key, value in state.items())
    if isinstance(state, (tuple, list)):
        return sys.getsizeof(state) + sum(_state_bytes(value) for value in state)
    if state is None or isinstance(state, (bool, int, float)):
        return sys.getsizeof(state)
    raise TypeError("Cache state must contain tensors, scalars, and standard containers")


class TorchSURTBackend(nn.Module):
    """Dual-path masks -> shared streaming Zipformer -> greedy stateless RNN-T.

    decode_tokens converts a sequence of token IDs to text. Text is returned
    only at endpoints/EOF so BPE word boundaries never depend on chunking.
    The decoder context is bounded; pending text tokens have an explicit cap.
    Encoder lengths count emitted frames, excluding padding and right context.
    """
    num_channels = 2
    chunk_frames = 32
    frame_shift_samples = 160

    def __init__(self, mask_estimator, encoder, decoder, joiner, decode_tokens, *,
                 blank_id: int, context_size: int = 2, right_context_frames: int = 0,
                 joint_encoder_layer=None,
                 encoder_subsampling: int = 4, max_state_bytes: int = 64 * 1024 * 1024,
                 endpoint_silence_ms: int = 800, max_pending_tokens: int = 8192,
                 max_symbols_per_frame: int = 4, device: str = "cpu"):
        super().__init__()
        if context_size < 1 or encoder_subsampling < 1 or max_symbols_per_frame < 1 or max_pending_tokens < 1:
            raise ValueError("Invalid decoder dimensions or bounds")
        if right_context_frames not in (0, 1) or max_state_bytes < 1 or endpoint_silence_ms < 1:
            raise ValueError("Invalid context, endpoint, or cache budget")
        self.mask_estimator, self.encoder = mask_estimator, encoder
        self.decoder, self.joiner = decoder, joiner
        self.joint_encoder_layer = joint_encoder_layer
        self.decode_tokens = decode_tokens
        self.blank_id, self.context_size = blank_id, context_size
        self.right_context_frames = right_context_frames
        self.encoder_subsampling = encoder_subsampling
        self.max_state_bytes = max_state_bytes
        self.endpoint_frames = (endpoint_silence_ms + 9) // 10
        self.max_pending_tokens = max_pending_tokens
        self.max_symbols_per_frame = max_symbols_per_frame
        self.device = torch.device(device)
        self.to(self.device).eval()
        self.reset()

    def _check_state(self):
        if _state_bytes((self.mask_state, self.encoder_state)) > self.max_state_bytes:
            raise RuntimeError("SURT acoustic cache exceeds its configured memory budget")

    @torch.inference_mode()
    def reset(self):
        self.mask_state = self.mask_estimator.initial_state()
        self.encoder_state = self.encoder.initial_state()
        self.contexts = [[self.blank_id] * self.context_size for _ in range(2)]
        self.tokens = [[], []]
        self.blanks = [0, 0]
        self.finished = False
        self._check_state()

    def _commit(self, channel):
        text = self.decode_tokens(self.tokens[channel]) if self.tokens[channel] else ""
        self.tokens[channel].clear()
        return text

    @torch.inference_mode()
    def step(self, features: np.ndarray, *, valid_frames: int, final: bool) -> BackendOutput:
        if self.finished:
            raise RuntimeError("Backend is finalized")
        if features.shape != (32 + self.right_context_frames, 80) or not 1 <= valid_frames <= 32:
            raise ValueError("Invalid fixed SURT feature block")
        x = torch.as_tensor(features, device=self.device).unsqueeze(0)
        masks, self.mask_state = self.mask_estimator.forward_chunk(x, self.mask_state, valid_frames, final)
        if masks.shape != (1, x.shape[1], 160) or not torch.isfinite(masks).all():
            raise ValueError("DPRNN must produce finite masks [1,W,80*2]")
        masks = masks.reshape(1, x.shape[1], 80, 2)
        separated = torch.cat([x * masks[..., channel] for channel in range(2)], dim=0)
        encoded, lengths, self.encoder_state = self.encoder.forward_chunk(
            separated, self.encoder_state, valid_frames, final)
        if encoded.ndim != 3 or encoded.shape[0] != 2 or len(lengths) != 2:
            raise ValueError("Zipformer must return [2,T,D] and two valid output lengths")
        if self.joint_encoder_layer is not None:
            if int(lengths[0]) != int(lengths[1]):
                raise ValueError("Branch tying requires aligned output channels")
            # Recur across the channel axis independently at each time index.
            encoded, _ = self.joint_encoder_layer(encoded)
        self._check_state()
        texts = []
        for channel in range(2):
            length = int(lengths[channel])
            if not 0 <= length <= encoded.shape[1]:
                raise ValueError("Invalid Zipformer output length")
            for frame in encoded[channel, :length]:
                emitted = False
                for _ in range(self.max_symbols_per_frame):
                    context = torch.tensor([self.contexts[channel]], dtype=torch.int64, device=self.device)
                    prediction = self.decoder(context, need_pad=False)
                    logits = self.joiner(frame.reshape(1, 1, 1, -1), prediction.reshape(1, 1, 1, -1))
                    token = int(logits.reshape(-1).argmax())
                    if token == self.blank_id:
                        break
                    emitted = True
                    self.tokens[channel].append(token)
                    if len(self.tokens[channel]) > self.max_pending_tokens:
                        raise RuntimeError("Pending transcript token budget exceeded before endpoint")
                    self.contexts[channel] = (self.contexts[channel] + [token])[-self.context_size:]
                self.blanks[channel] = 0 if emitted else self.blanks[channel] + self.encoder_subsampling
            endpoint = self.blanks[channel] >= self.endpoint_frames
            texts.append(self._commit(channel) if final or endpoint else "")
            if endpoint:
                self.contexts[channel] = [self.blank_id] * self.context_size
        return BackendOutput(tuple(texts), tuple(self.blanks))

    @torch.inference_mode()
    def finish(self) -> BackendOutput:
        texts = tuple(self._commit(channel) for channel in range(2))
        self.finished = True
        self.mask_state = self.encoder_state = None
        return BackendOutput(texts, tuple(self.blanks))


def load_torchscript_backend(config):
    """Load a trusted export bundle; never infer architecture from a state_dict.

    Required: metadata.json, mask.pt, encoder.pt, decoder.pt, joiner.pt,
    tokenizer.model. An optional branch_tying.pt implements the channel LSTM.
    All components must come from the same validated recipe/checkpoint export.
    """
    import json
    from pathlib import Path
    import sentencepiece as spm

    root = Path(config["checkpoint"])
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("architecture") != "surt2-dprnn-zipformer-stateless":
        raise ValueError("Export metadata must declare the SURT 2.0 architecture")
    if metadata.get("feature_profile") != "kaldi80-snip-pcm16-v1":
        raise ValueError("Export must be validated against this streaming feature profile")
    device = config.get("device", "cpu")
    def load(name):
        return torch.jit.load(str(root / name), map_location=device).eval()
    tokenizer = spm.SentencePieceProcessor(model_file=str(root / "tokenizer.model"))
    joint = load("branch_tying.pt") if metadata.get("branch_tying", False) else None
    return TorchSURTBackend(load("mask.pt"), load("encoder.pt"), load("decoder.pt"),
        load("joiner.pt"), tokenizer.decode, blank_id=int(metadata["blank_id"]),
        context_size=int(metadata["context_size"]),
        right_context_frames=int(metadata["right_context_frames"]),
        encoder_subsampling=int(metadata["encoder_subsampling"]),
        joint_encoder_layer=joint, max_state_bytes=int(metadata["max_state_bytes"]),
        endpoint_silence_ms=int(config.get("endpoint_silence_ms", 800)), device=device)
