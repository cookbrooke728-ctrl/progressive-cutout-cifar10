"""Download public Release weights, check SHA-256, and rerun the full test protocol."""
import argparse
import concurrent.futures
import json
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutout_lab.data import ResidentData
from cutout_lab.engine import evaluate_run
from cutout_lab.utils import dump_json, seed_everything, sha256

parser = argparse.ArgumentParser()
parser.add_argument("--repository-url", required=True)
args = parser.parse_args()
manifest = json.loads(Path("release-assets/manifest.json").read_text(encoding="utf-8"))
destination = Path(".tmp/download-verification")
destination.mkdir(parents=True, exist_ok=True)


def download(entry):
    run = destination / entry["run"]
    run.mkdir(exist_ok=True)
    path = run / "best.pt"
    url = f"{args.repository_url.rstrip('/')}/releases/download/v1.0.0/{entry['filename']}"
    if not path.exists() or sha256(path) != entry["sha256"]:
        with urllib.request.urlopen(url, timeout=120) as response, open(path.with_suffix(".download"), "wb") as stream:
            shutil.copyfileobj(response, stream)
        path.with_suffix(".download").replace(path)
    assert sha256(path) == entry["sha256"]
    shutil.copy2(Path("runs") / entry["run"] / "status.json", run / "status.json")
    return entry, run, url


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    downloaded = list(executor.map(download, manifest["weights"]))
seed_everything(42)
data = ResidentData()
results = []
for entry, run, url in downloaded:
    actual = evaluate_run(data, run, force=True)
    expected = json.loads((Path("runs") / entry["run"] / "evaluation.json").read_text(encoding="utf-8"))
    assert actual["weight_sha256"] == expected["weight_sha256"]
    for key in ("accuracy", "macro_f1"):
        assert abs(actual["clean"][key] - expected["clean"][key]) < 1e-8
    assert abs(actual["mean_occluded_accuracy"] - expected["mean_occluded_accuracy"]) < 1e-8
    results.append({"run": entry["run"], "download_url": url, "sha256": actual["weight_sha256"],
                    "clean_accuracy": actual["clean"]["accuracy"],
                    "mean_occluded_accuracy": actual["mean_occluded_accuracy"], "passed": True})
dump_json("output/download_verification.json", {"all_passed": True, "weights": results})
print("All 12 public weights reproduce the original metrics", flush=True)
