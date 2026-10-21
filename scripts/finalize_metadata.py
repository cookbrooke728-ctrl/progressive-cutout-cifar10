"""Populate publication URLs and README metrics after the complete-run audit."""
import argparse
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repository-url", required=True)
args = parser.parse_args()
url = args.repository_url.rstrip("/")
assert re.fullmatch(r"https://github\.com/[^/]+/[^/]+", url)
summary = json.loads(Path("output/summary.json").read_text(encoding="utf-8"))
audit = json.loads(Path("output/audit.json").read_text(encoding="utf-8"))
assert audit["complete_runs"] == 12 and audit["prediction_metrics_verified"]
publication_path = Path("paper/publication.json")
publication = json.loads(publication_path.read_text(encoding="utf-8"))
publication.update(repository_url=url, release_url=url + "/releases/tag/v1.0.0")
publication_path.write_text(json.dumps(publication, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = ["## Weights and release", "", f"Code: {url}", "", f"Weights and evidence: {publication['release_url']}", "",
         f"All 12 formal runs completed {summary['epochs']} epochs. Values below are mean ± sample SD over three training seeds, in percent.", "",
         "| Method | Clean accuracy | Macro-F1 | Mean occluded accuracy |", "|---|---:|---:|---:|"]
for method, result in summary["methods"].items():
    formatted = [f"{result[key]['mean']:.2f} ± {result[key]['std']:.2f}" for key in ("accuracy", "macro_f1", "mean_occluded_accuracy")]
    lines.append("| " + " | ".join([method, *formatted]) + " |")
lines.extend(["", "Every validation-selected weight is provided; no seed selection.", "",
              "| Run | Best epoch | Clean accuracy | Weight | SHA-256 |", "|---|---:|---:|---|---|"])
for run in summary["runs"]:
    name = f"{run['method']}_seed{run['seed']}"
    asset = f"{url}/releases/download/v1.0.0/{name}_best.pt"
    lines.append(f"| {name} | {run['best_epoch']} | {run['accuracy']:.2f} | [download]({asset}) | `{run['weight_sha256']}` |")
lines.extend(["", "`scripts/verify_downloaded_weights.py` downloads all 12 public weights, checks their SHA-256 and reruns the full clean/occluded test protocol.", "",
              "## Build the paper", "", "After `report`, install `python-docx` in a document environment and run `python paper/build_paper.py`.",
              "On Windows with Word installed, export with `scripts/render_with_word.ps1`; inspect the rendered PDF before submission.",
              "The paper builder only reads the audited summary; author and URLs are in `paper/publication.json`.", ""])
readme_path = Path("README.md")
readme = readme_path.read_text(encoding="utf-8")
start = readme.index("## Weights and release")
end = readme.index("## Data download")
readme_path.write_text(readme[:start] + "\n".join(lines) + "\n" + readme[end:], encoding="utf-8")
print(json.dumps(publication, ensure_ascii=False))
