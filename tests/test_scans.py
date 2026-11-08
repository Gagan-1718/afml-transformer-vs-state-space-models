"""Every scan backend must compute the same function, forward and backward."""
import pytest
import torch
import torch.nn.functional as F

from afml.models.mamba import (HAS_CUDA_SCAN, selective_scan_chunked, selective_scan_cuda,
                               selective_scan_parallel, selective_scan_sequential)


def inputs(b=2, l=37, d=6, n=4, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(b, l, d, generator=g)
    delta = F.softplus(torch.randn(b, l, d, generator=g))
    A = -torch.exp(torch.randn(d, n, generator=g))
    B, C, D = torch.randn(b, l, n, generator=g), torch.randn(b, l, n, generator=g), torch.randn(d, generator=g)
    return [t.to(device).requires_grad_() for t in (u, delta, A, B, C, D)]


def run(fn, args, **kw):
    y, h = fn(*args, **kw)
    (y.square().sum() + h.square().sum()).backward()
    grads = [a.grad.clone() for a in args]
    for a in args:
        a.grad = None
    return y.detach(), h.detach(), grads


@pytest.mark.parametrize("fn,kw", [
    (selective_scan_parallel, {}),
    (selective_scan_chunked, {"chunk_size": 8}),
    (selective_scan_chunked, {"chunk_size": 64}),  # one chunk longer than the sequence
])
def test_backend_matches_sequential_forward_and_backward(fn, kw):
    args = inputs()  # l = 37 is deliberately not a multiple of the chunk size
    y0, h0, g0 = run(selective_scan_sequential, args)
    y1, h1, g1 = run(fn, args, **kw)
    torch.testing.assert_close(y1, y0, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(h1, h0, atol=1e-5, rtol=1e-4)
    for a, b in zip(g1, g0):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-3)


def test_initial_state_is_respected():
    u, delta, A, B, C, D = [t.detach() for t in inputs(l=20)]
    y_full, h_full = selective_scan_sequential(u, delta, A, B, C, D)
    _, h_mid = selective_scan_sequential(u[:, :9], delta[:, :9], A, B[:, :9], C[:, :9], D)
    for fn, kw in [(selective_scan_sequential, {}), (selective_scan_parallel, {}),
                   (selective_scan_chunked, {"chunk_size": 4})]:
        y2, h2 = fn(u[:, 9:], delta[:, 9:], A, B[:, 9:], C[:, 9:], D, h0=h_mid, **kw)
        torch.testing.assert_close(y2, y_full[:, 9:], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(h2, h_full, atol=1e-5, rtol=1e-4)


def test_chunked_does_not_store_full_state_tensor():
    """The point of the chunked backend: saved activations scale with l*d, not l*d*n*log(l)."""
    saved = []

    def pack(t):
        saved.append(t.numel())
        return t

    b, l, d, n = 1, 512, 32, 16
    args = inputs(b, l, d, n)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y, h = selective_scan_chunked(*args, chunk_size=32)
    assert max(saved) < b * l * d * n  # never keeps a (b, l, d, n) tensor
    saved.clear()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        selective_scan_parallel(*args)
    assert sum(saved) > 4 * b * l * d * n  # the parallel scan keeps many of them


@pytest.mark.skipif(not (HAS_CUDA_SCAN and torch.cuda.is_available()), reason="needs mamba-ssm and a GPU")
def test_cuda_kernel_matches_reference():
    args = inputs(b=2, l=300, d=16, n=8, device="cuda")
    y0, h0, g0 = run(selective_scan_sequential, args)
    y1, h1, g1 = run(selective_scan_cuda, args)
    torch.testing.assert_close(y1, y0, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(h1, h0, atol=1e-3, rtol=1e-3)
    for a, b in zip(g1, g0):
        torch.testing.assert_close(a, b, atol=1e-2, rtol=1e-2)


def _fake_mamba_pkg(root, interface_body):
    pkg = root / "mamba_ssm"
    for sub in ("", "ops", "utils"):
        (pkg / sub).mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("raise ImportError('the full package import is broken')\n")
    (pkg / "ops" / "__init__.py").write_text("")
    (pkg / "utils" / "__init__.py").write_text("")
    (pkg / "utils" / "torch.py").write_text("HELPER = 'helper'\n")
    (pkg / "ops" / "selective_scan_interface.py").write_text(interface_body)


@pytest.mark.parametrize("works", [True, False])
def test_cuda_scan_loader_skips_package_init(tmp_path, monkeypatch, works):
    import importlib
    import sys

    from afml.models.mamba import _load_cuda_scan
    body = ("from mamba_ssm.utils.torch import HELPER\n"
            "def selective_scan_fn(*args, **kwargs):\n    return HELPER\n") if works else \
        "import selective_scan_cuda_that_does_not_exist\n"
    _fake_mamba_pkg(tmp_path, body)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.split(".")[0] == "mamba_ssm"]:
        monkeypatch.delitem(sys.modules, name)
    importlib.invalidate_caches()
    fn, err = _load_cuda_scan()
    if works:
        assert err is None and fn() == "helper"  # loaded although mamba_ssm/__init__.py raises
    else:
        assert fn is None and "selective_scan_cuda_that_does_not_exist" in err
        assert not [m for m in sys.modules if m.split(".")[0] == "mamba_ssm"]  # cleaned up
    for name in [m for m in sys.modules if m.split(".")[0] == "mamba_ssm"]:
        del sys.modules[name]
