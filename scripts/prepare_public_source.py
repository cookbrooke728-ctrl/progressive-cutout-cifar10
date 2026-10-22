"""Export only project deliverables, without local Git history or credentials."""
import argparse
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--destination", default=".tmp/public-source")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
destination = (root / args.destination).resolve()
assert destination.is_relative_to((root / ".tmp").resolve())
audit = json.loads((root / "output/audit.json").read_text(encoding="utf-8"))
assert audit["complete_runs"] == 12 and audit["prediction_metrics_verified"]
destination.mkdir(parents=True, exist_ok=True)
files = [root / name for name in (".gitignore", ".gitattributes", "README.md", "LICENSE", "pyproject.toml", "requirements.txt", "requirements-lock.txt")]
for folder in (".github", "cutout_lab", "configs", "tests", "scripts", "paper", "runs", "output", "validation"):
    for path in (root / folder).rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in ("__pycache__", "qa", "gpu_resume_direct", "gpu_resume_split") for part in rel.parts):
            continue
        if path.suffix in (".pt", ".pyc", ".tmp", ".log", ".zip"):
            continue
        if path.name in ("install_gh.py", "gh_with_proxy.py"):
            continue
        files.append(path)
files.extend(root / "data" / name for name in ("split.json", "normalization.json", "download_provenance.json"))
ready = root / ".release-ready"
if ready.exists():
    files.append(ready)
for path in files:
    target = destination / path.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
print(json.dumps({"destination": str(destination), "files": len(files), "bytes": sum(p.stat().st_size for p in files)}))
