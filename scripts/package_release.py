"""Build release assets only from complete, audited formal experiments."""
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def add_canonical_file(archive, path, name):
    # Byte-identical on Windows and Linux; timestamps are fixed for reproducibility.
    entry = zipfile.ZipInfo(str(name).replace("\\", "/"), date_time=(1980, 1, 1, 0, 0, 0))
    entry.create_system = 3
    entry.external_attr = 0o100644 << 16
    entry.compress_type = zipfile.ZIP_STORED
    archive.writestr(entry, Path(path).read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-url", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / "output/audit.json").read_text(encoding="utf-8"))
    assert audit["complete_runs"] == 12 and audit["prediction_metrics_verified"]
    destination = root / "release-assets"
    destination.mkdir(exist_ok=True)
    entries = []
    for row in audit["artifacts"]:
        name = row["run"]
        run = root / "runs" / name
        assert sha256(run / "best.pt") == row["weight_sha256"]
        target = destination / f"{name}_best.pt"
        shutil.copy2(run / "best.pt", target)
        evaluation = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
        entries.append({"run": name, "filename": target.name, "sha256": sha256(target),
                        "best_epoch": evaluation["best_epoch"], "clean_accuracy": evaluation["clean"]["accuracy"],
                        "mean_occluded_accuracy": evaluation["mean_occluded_accuracy"]})
    evidence = destination / "experiment_evidence.zip"
    with zipfile.ZipFile(evidence, "w", zipfile.ZIP_STORED) as archive:
        for base in (root / "runs", root / "output", root / "configs"):
            for path in sorted(base.rglob("*")):
                if path.is_file() and path.suffix not in (".pt", ".tmp", ".zip") and "qa" not in path.parts:
                    add_canonical_file(archive, path, path.relative_to(root))
        for name in ("split.json", "normalization.json", "download_provenance.json"):
            path = root / "data" / name
            if path.exists():
                add_canonical_file(archive, path, path.relative_to(root))
    for extension in ("docx", "pdf"):
        paper = root / "output/paper" / f"progressive_cutout_paper.{extension}"
        if not paper.is_file():
            raise FileNotFoundError(f"Final paper missing: {paper}")
        shutil.copy2(paper, destination / paper.name)
    dump_json(destination / "manifest.json", {"repository_url": args.repository_url, "tag": "v1.0.0",
                                             "weights": entries, "evidence_sha256": sha256(evidence)})
    summary = json.loads((root / "output/summary.json").read_text(encoding="utf-8"))
    notes = ["# CIFAR-10 Cutout reproduction v1.0.0", "", f"All 12 runs trained locally for {summary['epochs']} epochs, seeds 42/43/44.",
             "The release workflow only validates and packages existing files; it performs no model training.", "",
             "| Method | Clean accuracy (%) | Mean occluded accuracy (%) |", "|---|---:|---:|"]
    for method, result in summary["methods"].items():
        notes.append(f"| {method} | {result['accuracy']['mean']:.2f} ± {result['accuracy']['std']:.2f} | {result['mean_occluded_accuracy']['mean']:.2f} ± {result['mean_occluded_accuracy']['std']:.2f} |")
    notes.extend(["", "All validation-selected weights, hashes, original logs, per-example predictions and the Word/PDF paper are included.", "", args.repository_url])
    (destination / "RELEASE_NOTES.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    with open(destination / "SHA256SUMS.txt", "w", encoding="utf-8") as stream:
        for path in sorted(destination.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS.txt":
                stream.write(f"{sha256(path)}  {path.name}\n")
    print(f"Packaged {len(entries)} weight files and evidence archive in {destination}")


if __name__ == "__main__":
    main()
