"""1C-B adapters for the exact archived SURT recipe; inference only, B=1.

No parameter copies or new modules. Only the bidirectional *row* LSTM resets
per segment, exactly as in upstream. Column, Zipformer and joint states persist.
"""
import torch
from frontend import BufferedFbank


def owned(x):
    return x.detach().clone()


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple, dict)):
        for item in (value.values() if isinstance(value, dict) else value):
            yield from tensors(item)


def state_bytes(value):
    # Storage, not numel: a short view must not hide retained acoustic history.
    seen = {}
    for x in tensors(value):
        storage = x.untyped_storage()
        seen[(str(x.device), storage.data_ptr())] = storage.nbytes()
    return sum(seen.values())


class MaskStream:
    def __init__(self, module):
        self.m = module
        self.s = module.segment_size
        if self.s != 32 or module.training or module.chunk_width_randomization:
            raise RuntimeError('Expected reviewed eval DPRNN segment=32, no randomization')
        self.h = self.s // 2
        self.reset()

    @torch.inference_mode()
    def reset(self):
        p = next(self.m.parameters())
        self.buffer = p.new_zeros(1, self.h, self.m.input_size)
        self.raw = p.new_empty(1, 0, 80)
        self.columns = [None] * len(self.m.col_rnn)
        self.overlap = None
        self.total = self.emitted = 0
        self.finished = False

    def segment(self, x):
        # x: B=1,S,D. Column batch is the fixed within-segment position S.
        for i, (row, col) in enumerate(zip(self.m.row_rnn, self.m.col_rnn)):
            x = row(x)
            z = x.transpose(0, 1)  # S,1,D; batch_first=True
            z_lstm, state = col.lstm(z, self.columns[i])
            self.columns[i] = tuple(owned(t) for t in state)
            x = col.norm_final(col.balancer(col.dropout(z_lstm) + z)).transpose(0, 1)
        return x

    @torch.inference_mode()
    def push(self, x, final=False):
        if self.finished:
            raise RuntimeError('DPRNN finalized')
        if x.shape[0] != 1 or x.shape[1] > 32 or x.shape[2] != 80:
            raise ValueError('DPRNN bounded input: 1,T<=32,80')
        self.total += x.shape[1]
        self.raw = torch.cat((self.raw, x), 1)
        self.buffer = torch.cat((self.buffer, self.m.input_embed(x)), 1) if x.shape[1] else self.buffer
        result = []
        while self.buffer.shape[1] >= self.s or (final and self.emitted < self.total):
            if self.buffer.shape[1] < self.s:
                self.buffer = torch.cat((self.buffer, self.buffer.new_zeros(
                    1, self.s - self.buffer.shape[1], self.m.input_size)), 1)
            y = self.segment(self.buffer[:, :self.s])
            self.buffer = owned(self.buffer[:, self.h:])
            if self.overlap is not None:
                count = min(self.h, self.total - self.emitted)
                mask = torch.relu(self.m.out_embed(self.overlap + y[:, :self.h]))[:, :count]
                masks = mask.view(1, count, 80, 2).unbind(-1)
                raw = self.raw[:, :count]
                result.append(torch.cat([raw * m for m in masks], 0))
                self.raw = owned(self.raw[:, count:])
                self.emitted += count
            self.overlap = owned(y[:, self.h:])
        if final:
            self.finished = True
            self.buffer = self.buffer[:, :0].clone()
            self.overlap = None
            assert self.emitted == self.total and self.raw.shape[1] == 0
        return torch.cat(result, 1) if result else x.new_empty(2, 0, 80)


class RecognitionStream:
    def __init__(self, model):
        self.m = model
        self.stride = 2 * model.encoder.decode_chunk_size
        if self.stride != 32 or model.num_channels != 2:
            raise RuntimeError('Expected reviewed two-branch 32-frame contract')
        core = model.joint_encoder_layer[1]
        if not core.batch_first or core.bidirectional:
            raise RuntimeError('Expected causal batch-first branch LSTM')
        self.reset()

    @torch.inference_mode()
    def reset(self):
        p = next(self.m.parameters())
        # Upstream initializes B=1. Keep independent cache sets for each branch.
        self.zip_states = [self.m.encoder.get_init_state(p.device) for _ in range(2)]
        self.pending = p.new_empty(2, 0, 80)
        self.joint = None
        self.total = self.emitted = 0
        self.context = [[-1] * (self.m.decoder.context_size - 1) +
                        [self.m.decoder.blank_id] for _ in range(2)]
        self.prediction = self.predict()
        self.finished = False

    def predict(self):
        y = torch.tensor(self.context, device=next(self.m.parameters()).device, dtype=torch.int64)
        return self.m.joiner.decoder_proj(self.m.decoder(y, need_pad=False))

    def decode(self, x):
        events = []
        am = self.m.joiner.encoder_proj(x)
        blank = self.m.decoder.blank_id
        unk = getattr(self.m, 'unk_id', blank)  # exact upstream greedy policy
        for t in range(x.shape[1]):
            logits = self.m.joiner(am[:, t:t+1].unsqueeze(1),
                                   self.prediction.unsqueeze(1), project_input=False)
            ids = logits[:, 0, 0].argmax(-1).tolist()
            changed = False
            for branch, token in enumerate(ids):
                if token not in (blank, unk):
                    self.context[branch] = (self.context[branch] + [token])[-self.m.decoder.context_size:]
                    events.append(dict(branch=branch, token=token, frame=self.emitted + t))
                    changed = True
            if changed:
                self.prediction = self.predict()
        return events

    @torch.inference_mode()
    def push(self, x, final=False):
        if self.finished:
            raise RuntimeError('Recognition finalized')
        if x.shape[0] != 2 or x.shape[1] > 32:
            raise ValueError('Recognition bounded input: 2,T<=32,80')
        self.total += x.shape[1]
        self.pending = torch.cat((self.pending, x), 1)
        target = (max(0, (self.total - 7) // 2) + 1) // 2
        outputs, events = [], []
        while self.pending.shape[1] >= self.stride + 7 or (final and self.emitted < target):
            piece = self.pending[:, :self.stride + 7]
            if piece.shape[1] < self.stride + 7:
                # Explicit same-context EOF policy, after masks, never before DPRNN.
                piece = torch.cat((piece, piece.new_full(
                    (2, self.stride + 7 - piece.shape[1], 80), -23.025850929940457)), 1)
            branches = []
            for branch in range(2):
                y, lens, states = self.m.encoder.streaming_forward(
                    piece[branch:branch+1], torch.tensor([39], device=piece.device), self.zip_states[branch])
                assert lens.item() == 8 and y.shape[1] == 8
                self.zip_states[branch] = [owned(t) for t in states]
                branches.append(y)
            self.pending = owned(self.pending[:, self.stride:])
            count = min(8, target - self.emitted) if final else 8
            encoded = torch.cat(branches, 0)[:, :count]
            wrapper = self.m.joint_encoder_layer
            tied, self.joint = wrapper[1](wrapper[0](encoded), self.joint)
            self.joint = tuple(owned(t) for t in self.joint)
            tied = wrapper[4](wrapper[3](tied))
            events.extend(self.decode(tied))
            self.emitted += count
            outputs.append(tied)
        if final:
            self.finished = True
            self.pending = self.pending[:, :0].clone()
            assert self.emitted == target
        dim = self.m.joint_encoder_layer[1].input_size // 2
        return (torch.cat(outputs, 1) if outputs else x.new_empty(2, 0, dim)), events


class NativeSURT:
    """Push <=3200 samples. Outputs/events belong to caller; no transcript history."""
    def __init__(self, model, extractor):
        if model.training:
            raise ValueError('Use eval model')
        self.model = model
        self.frontend = BufferedFbank(extractor, 3200)
        self.mask = MaskStream(model.mask_encoder)
        self.recognition = RecognitionStream(model)
        self.finished = False
        self.peak_state_bytes = 0

    @torch.inference_mode()
    def reset(self):
        self.frontend.reset()
        self.mask.reset()
        self.recognition.reset()
        self.finished = False
        self.peak_state_bytes = 0

    def acoustic_bytes(self):
        values = [self.mask.buffer, self.mask.raw, self.mask.columns, self.mask.overlap,
                  self.recognition.pending, self.recognition.zip_states, self.recognition.joint,
                  self.recognition.prediction]
        return state_bytes(values) + self.frontend.buffer.nbytes

    @torch.inference_mode()
    def push(self, samples, final=False):
        if self.finished:
            if final and len(samples) == 0:
                p = next(self.model.parameters())
                return p.new_empty(2, 0, self.model.joint_encoder_layer[1].input_size // 2), []
            raise RuntimeError('Stream finalized; reset before reuse')
        features = self.frontend.push(samples, final=final)
        x = torch.as_tensor(features, device=next(self.model.parameters()).device).unsqueeze(0)
        outputs, events = [], []
        # Bound intermediate acoustic tensors even when feature packet boundaries differ.
        for piece in x.split(16, dim=1):
            if not piece.shape[1]:
                continue
            masked = self.mask.push(piece)
            for part in masked.split(16, dim=1):
                y, emitted = self.recognition.push(part)
                outputs.append(y)
                events.extend(emitted)
            self.peak_state_bytes = max(self.peak_state_bytes, self.acoustic_bytes())
        if final:
            masked = self.mask.push(x[:, :0], final=True)
            for part in masked.split(16, dim=1):
                y, emitted = self.recognition.push(part)
                outputs.append(y)
                events.extend(emitted)
            y, emitted = self.recognition.push(masked[:, :0], final=True)
            outputs.append(y)
            events.extend(emitted)
            self.finished = True
        self.peak_state_bytes = max(self.peak_state_bytes, self.acoustic_bytes())
        p = next(self.model.parameters())
        return (torch.cat(outputs, 1) if outputs else p.new_empty(
            2, 0, self.model.joint_encoder_layer[1].input_size // 2)), events
