import json
from pathlib import Path

import torch

from ..config import ModelConfig
from ..models import build_model


def device_auto(device="auto"):
    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


def load_model(run_dir, which="best", device="auto", **overrides):
    """Load best.pt (or last.pt) from a training run. overrides e.g. scan='chunked'."""
    device = device_auto(device)
    state = torch.load(Path(run_dir) / f"{which}.pt", map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict({**state["model_config"], **overrides})
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def run_label(run_dir):
    s = json.loads((Path(run_dir) / "summary.json").read_text())
    arch = s["arch"] if s["arch"] == "transformer" else ("mamba" if s["selective"] else "mamba_lti")
    return arch, s


class Timer:
    """Wall-clock timing that is correct on GPUs (CUDA events) and CPUs."""

    def __init__(self, device):
        self.cuda = device == "cuda"

    def __enter__(self):
        if self.cuda:
            self.a, self.b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            self.a.record()
        else:
            import time
            self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            self.b.record()
            torch.cuda.synchronize()
            self.ms = self.a.elapsed_time(self.b)
        else:
            import time
            self.ms = (time.perf_counter() - self.t) * 1000
