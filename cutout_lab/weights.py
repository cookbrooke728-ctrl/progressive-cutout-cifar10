"""One checked weight resolver for evaluation, reporting and restoration."""
import json
import shutil
from pathlib import Path
import torch
from .utils import sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_weight(run, weights=None):
    run = Path(run)
    config, status = read_json(run / "config.json"), read_json(run / "status.json")
    method, seed = config["method"], config["seed"]
    if run.name != f"{method}_seed{seed}" or status["method"] != method or status["seed"] != seed:
        raise ValueError(f"Method/seed metadata mismatch: {run}")
    expected = status["best_weight_sha256"]
    local = run / "best.pt"
    if local.is_file() and weights is None:
        path = local
    else:
        folder = Path(weights) if weights is not None else run.parent.parent / "weights"
        manifest = folder / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing {local}; published manifest not found: {manifest}. Use restore or supply weights/.")
        entries = [e for e in read_json(manifest)["weights"] if e["run"] == run.name]
        if len(entries) != 1:
            raise ValueError(f"Manifest requires exactly one entry for {run.name}")
        entry = entries[0]
        if entry["filename"] != f"{run.name}_best.pt" or entry["sha256"] != expected:
            raise ValueError(f"Manifest identity/SHA-256 mismatch: {run.name}")
        path = folder / entry["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing weight for {method}, seed {seed}: {path}")
    if sha256(path) != expected:
        raise ValueError(f"Weight SHA-256 mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != config or checkpoint["best_epoch"] != status["best_epoch"]:
        raise ValueError(f"Checkpoint configuration/method/seed/epoch mismatch: {path}")
    return path


def restore_runs(source="runs", weights="weights", destination="verification/runs"):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Destination must be new; original evidence is never overwritten: {destination}")
    runs = sorted(p for p in source.glob("*_seed*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run metadata found: {source}")
    entries = read_json(Path(weights) / "manifest.json")["weights"]
    names = [e["run"] for e in entries]
    if len(names) != len(set(names)) or set(names) != {r.name for r in runs}:
        raise ValueError("Manifest/run set mismatch: missing or duplicate run metadata")
    # Validate the whole batch before creating any output. Always validate the published copy.
    verified = []
    for run in runs:
        verified.append((run, resolve_weight(run, weights)))
    destination.mkdir(parents=True)
    for run, weight in verified:
        target = destination / run.name
        target.mkdir()
        for name in ("config.json", "status.json", "provenance.json", "initialization.json", "history.csv"):
            shutil.copyfile(run / name, target / name)
        shutil.copyfile(weight, target / "best.pt")
    return [r.name for r, _ in verified]


def suite_action(run, resume=False):
    run = Path(run)
    status = run / "status.json"
    if status.exists() and read_json(status)["complete"]:
        resolve_weight(run)
        return "skip_verified_complete"
    if resume:
        if not (run / "last.pt").is_file():
            raise FileNotFoundError(f"Resume requires last.pt: {run}")
        return "resume"
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Existing incomplete run: {run}. Use --resume only with last.pt, or a new --output.")
    return "train"
