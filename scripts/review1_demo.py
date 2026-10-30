"""Run from the repository root: python scripts/review1_demo.py --help."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Small Transformer demo for AFML Review 1")
    parser.add_argument("action", choices=["prepare", "train", "generate"])
    parser.add_argument("--config", default="configs/review1.json")
    parser.add_argument("--data-dir", default="data/review1")
    parser.add_argument("--run-dir", default="runs/review1")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prompt", default="The purpose of education is")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.action == "prepare":
        from afml.data import prepare_data
        result = prepare_data(cfg["data"], args.data_dir)
        print(json.dumps({k: result[k] for k in ("revision", "vocab_size", "train", "validation")}, indent=2))
    elif args.action == "train":
        from afml.demo import train_demo
        result = train_demo(cfg["training"], args.data_dir, args.run_dir, args.device)
        print(f"Saved results to {args.run_dir}; best validation perplexity = {result['best_validation_perplexity']:.2f}")
    else:
        import torch
        from tokenizers import Tokenizer
        from afml.models import ModelConfig, build_model
        from afml.demo import generate
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(Path(args.run_dir) / "best.pt", map_location=device, weights_only=True)
        model = build_model(ModelConfig(**checkpoint["model_config"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        tokenizer = Tokenizer.from_file(str(Path(args.run_dir) / "tokenizer.json"))
        print(generate(model, tokenizer, args.prompt, device))


if __name__ == "__main__":
    main()
