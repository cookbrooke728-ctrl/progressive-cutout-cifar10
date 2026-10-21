import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Cutout reproduction with traceable results")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", default="data")
    train = sub.add_parser("train")
    train.add_argument("--method", choices=("baseline", "cutout", "matched", "progressive"), required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--stop-after", type=int)
    train.add_argument("--output", default="runs")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--device", default="cuda")
    train.add_argument("--data", default="data")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--runs", default="runs")
    evaluation.add_argument("--data", default="data")
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--force", action="store_true")
    evaluation.add_argument("--run")
    suite = sub.add_parser("suite")
    suite.add_argument("--config", default="configs/experiment.json")
    suite.add_argument("--stop-after-runs", type=int)
    report = sub.add_parser("report")
    report.add_argument("--runs", default="runs")
    report.add_argument("--output", default="output")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        _, _, split, stats = prepare(args.data, download=True)
        print(json.dumps({"train": len(split["train"]), "validation": len(split["validation"]), "normalization": stats}))
    elif args.command == "train":
        from .data import ResidentData
        from .engine import train_run
        data = ResidentData(args.data, args.device)
        train_run(data, args.method, args.seed, args.epochs, args.batch_size, args.output, args.resume, args.stop_after)
    elif args.command == "evaluate":
        from .data import ResidentData
        from .engine import evaluate_run
        from .utils import seed_everything
        seed_everything(42)
        data = ResidentData(args.data, args.device)
        for run in ([Path(args.run)] if args.run else sorted(Path(args.runs).glob("*_seed*"))):
            evaluate_run(data, run, args.force)
    elif args.command == "suite":
        from .data import ResidentData
        from .engine import train_run
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        data = ResidentData(config["data"], config.get("device", "cuda"))
        completed = 0
        for seed in config["seeds"]:
            for method in config["methods"]:
                run = Path(config["output"]) / f"{method}_seed{seed}"
                status = run / "status.json"
                if status.exists() and json.loads(status.read_text())["complete"]:
                    continue
                train_run(data, method, seed, config["epochs"], config["batch_size"], config["output"],
                          resume=(run / "last.pt").exists())
                completed += 1
                if args.stop_after_runs and completed >= args.stop_after_runs:
                    return
    elif args.command == "report":
        from .report import build_report
        build_report(args.runs, args.output)


if __name__ == "__main__":
    main()
