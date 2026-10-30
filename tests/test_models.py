import math

import pytest
import torch
import torch.nn.functional as F

from afml.models import ModelConfig, build_model
from afml.models.mamba import MambaBlock, selective_scan_parallel, selective_scan_sequential


def tiny(**kw) -> ModelConfig:
    base = dict(vocab_size=64, d_model=32, n_layers=2, max_seq_len=64, n_heads=4, d_state=8)
    base.update(kw)
    return ModelConfig(**base)


def test_parallel_scan_matches_sequential():
    torch.manual_seed(0)
    b, l, d, n = 2, 37, 6, 4  # l is deliberately not a power of two
    u = torch.randn(b, l, d)
    delta = F.softplus(torch.randn(b, l, d))
    A = -torch.exp(torch.randn(d, n))
    B, C, D = torch.randn(b, l, n), torch.randn(b, l, n), torch.randn(d)
    torch.testing.assert_close(
        selective_scan_parallel(u, delta, A, B, C, D),
        selective_scan_sequential(u, delta, A, B, C, D),
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.mark.parametrize("selective", [True, False])
def test_recurrent_step_matches_parallel_forward(selective):
    torch.manual_seed(0)
    block = MambaBlock(tiny(arch="mamba", selective=selective)).eval()
    x = torch.randn(2, 11, 32)
    with torch.no_grad():
        full = block(x)
        state = block.init_state(2)
        outs = []
        for t in range(x.shape[1]):
            y, state = block.step(x[:, t], state)
            outs.append(y)
    torch.testing.assert_close(full, torch.stack(outs, dim=1), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("arch", ["transformer", "mamba"])
def test_causality(arch):
    torch.manual_seed(0)
    model = build_model(tiny(arch=arch)).eval()
    idx = torch.randint(0, 64, (1, 20))
    idx2 = idx.clone()
    idx2[:, 10:] = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        l1, l2 = model(idx)[0], model(idx2)[0]
    torch.testing.assert_close(l1[:, :10], l2[:, :10], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("arch,selective", [("transformer", True), ("mamba", True), ("mamba", False)])
def test_overfits_a_tiny_batch(arch, selective):
    torch.manual_seed(0)
    model = build_model(tiny(arch=arch, selective=selective))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    data = torch.randint(0, 64, (4, 33))
    x, y = data[:, :-1], data[:, 1:]
    first = None
    for _ in range(150):
        _, loss = model(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert first == pytest.approx(math.log(64), abs=0.3)  # starts near uniform
    assert loss.item() < 0.5 * first
