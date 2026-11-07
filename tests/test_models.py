import math

import pytest
import torch

from afml.models import ModelConfig, build_model, generate, param_groups
from afml.models.mamba import MambaBlock


def tiny(**kw) -> ModelConfig:
    base = dict(vocab_size=64, d_model=32, n_layers=2, max_seq_len=64, n_heads=4, d_state=8, scan="chunked",
                chunk_size=8)
    base.update(kw)
    return ModelConfig(**base)


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
def test_cached_decoding_matches_full_forward(arch, selective):
    """Prefill + step-by-step decode (KV cache / recurrent state) must equal one full forward pass,
    including positions past max_seq_len for the Transformer's RoPE."""
    torch.manual_seed(0)
    model = build_model(tiny(arch=arch, selective=selective, max_seq_len=16)).eval()
    idx = torch.randint(0, 64, (2, 24))
    with torch.no_grad():
        full = model(idx)[0]
        caches = model.init_cache(2, 24)
        steps = [model.prefill(idx[:, :5], caches)]
        for t in range(5, 24):
            steps.append(model.decode_step(idx[:, t], caches))
    torch.testing.assert_close(torch.stack(steps[:-1], 1), full[:, 4:-1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(steps[-1], full[:, -1], atol=1e-4, rtol=1e-4)


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


def test_weight_decay_skips_ssm_parameters():
    model = build_model(tiny(arch="mamba"))
    decay, no_decay = param_groups(model, 0.1)
    names = {id(p): n for n, p in model.named_parameters()}
    no_decay_names = {names[id(p)] for p in no_decay["params"]}
    assert any(n.endswith("A_log") for n in no_decay_names)
    assert any(n.endswith(".D") for n in no_decay_names)
    assert any(n.endswith("dt_proj.bias") for n in no_decay_names)
    assert "embed.weight" in no_decay_names
    assert all(p.ndim >= 2 for p in decay["params"])  # linear and conv weights only
    total = sum(p.numel() for g in (decay, no_decay) for p in g["params"])
    assert total == sum(p.numel() for p in model.parameters())


def test_generate_shapes_and_determinism():
    model = build_model(tiny(arch="mamba"))
    idx = torch.randint(0, 64, (2, 5))
    a = generate(model, idx, 7, generator=torch.Generator().manual_seed(1))
    b = generate(model, idx, 7, generator=torch.Generator().manual_seed(1))
    assert a.shape == (2, 12) and torch.equal(a, b) and torch.equal(a[:, :5], idx)


def test_scan_backends_give_same_model_output():
    torch.manual_seed(0)
    cfg = tiny(arch="mamba")
    model = build_model(cfg).eval()
    idx = torch.randint(0, 64, (2, 30))
    outs = []
    for scan in ("sequential", "parallel", "chunked"):
        for layer in model.layers:
            layer.block.scan_name = scan
        with torch.no_grad():
            outs.append(model(idx)[0])
    for o in outs[1:]:
        torch.testing.assert_close(o, outs[0], atol=1e-5, rtol=1e-4)
