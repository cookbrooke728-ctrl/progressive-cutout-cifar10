import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import torch
from test_experiment import TinyData
from cutout_lab.engine import train_run
from cutout_lab.utils import dump_json

data = TinyData()
data.device = torch.device("cuda")
for name in ("padded", "labels", "validation", "validation_labels", "probe", "probe_labels"):
    setattr(data, name, getattr(data, name).cuda())
direct, h1 = train_run(data, "progressive", 42, 3, 8, run_dir="validation/gpu_resume_direct")
train_run(data, "progressive", 42, 3, 8, stop_after=1, run_dir="validation/gpu_resume_split")
resumed, h2 = train_run(data, "progressive", 42, 3, 8, resume=True, run_dir="validation/gpu_resume_split")
differences = {k: (v.float() - resumed.state_dict()[k].float()).abs().max().item() for k, v in direct.state_dict().items()}
result = {"maximum_absolute_tensor_difference": max(differences.values()), "epochs": 3,
          "probabilities": [r["p"] for r in h2], "passed": max(differences.values()) < 1e-5,
          "purpose": "synthetic pipeline validation only, not formal experiment evidence"}
dump_json("validation/gpu_resume.json", result)
print(json.dumps(result), flush=True)
assert result["passed"]
