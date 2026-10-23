"""Optional tensor-level composition tests; toy modules are not ASR models."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from streaming_asr.models.torch_surt import TorchSURTBackend


class Masks(nn.Module):
    def initial_state(self): return torch.zeros(1)
    def forward_chunk(self, x, state, valid_frames, final):
        return torch.ones((1, x.shape[1], 160), device=x.device), state + valid_frames


class Encoder(nn.Module):
    def initial_state(self): return torch.zeros(1)
    def forward_chunk(self, x, state, valid_frames, final):
        return x[:, :valid_frames:4], torch.tensor([(valid_frames + 3) // 4] * 2), state + valid_frames


class Predictor(nn.Module):
    def forward(self, context, need_pad=False):
        return context[:, -1:].float().unsqueeze(-1)


class Joiner(nn.Module):
    def forward(self, encoder, decoder):
        # Emit a token once after a blank context, then terminate with blank.
        token = int(decoder.flatten()[0]) == 0
        return torch.tensor([[[[0., 1.]]]]) if token else torch.tensor([[[[1., 0.]]]])


def backend(**kwargs):
    return TorchSURTBackend(Masks(), Encoder(), Predictor(), Joiner(),
                            lambda tokens: " ".join(map(str, tokens)), blank_id=0, **kwargs)


def test_masks_encoder_predictor_joiner_flush_and_bounded_context():
    model = backend()
    output = model.step(np.ones((32, 80), np.float32), valid_frames=32, final=False)
    assert output.texts == ("", "")
    assert model.contexts == [[0, 1], [0, 1]]
    final = model.finish()
    assert final.texts == ("1", "1")
    assert model.mask_state is model.encoder_state is None
    with pytest.raises(RuntimeError):
        model.step(np.ones((32, 80), np.float32), valid_frames=32, final=False)


def test_oversized_acoustic_state_is_rejected():
    class GrowingEncoder(Encoder):
        def forward_chunk(self, x, state, valid_frames, final):
            return x[:, :1], torch.tensor([1, 1]), torch.zeros(10000)
    model = backend(max_state_bytes=1024)
    model.encoder = GrowingEncoder()
    with pytest.raises(RuntimeError, match="cache"):
        model.step(np.ones((32, 80), np.float32), valid_frames=32, final=False)


def test_short_final_block_excludes_padding_and_flushes_tokens():
    model = backend()
    final = model.step(np.ones((32, 80), np.float32), valid_frames=1, final=True)
    assert final.texts == ("1", "1")
    assert model.finish().texts == ("", "")
