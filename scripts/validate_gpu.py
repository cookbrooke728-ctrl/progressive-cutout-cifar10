"""A small real-data overfit test, separate from official runs and test set."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from cutout_lab.data import prepare
from cutout_lab.model import ResNet20
from cutout_lab.utils import dump_json, seed_everything

seed_everything(7)
source, _, split, stats = prepare("data")
ids = split["probe"][:256]
x = torch.tensor(source.data[ids].transpose(0, 3, 1, 2).copy(), device="cuda").float() / 255
mean = torch.tensor(stats["mean"], device="cuda")[None, :, None, None]
std = torch.tensor(stats["std"], device="cuda")[None, :, None, None]
x = ((x - mean) / std).contiguous(memory_format=torch.channels_last)
y = torch.tensor([source.targets[i] for i in ids], device="cuda")
model = ResNet20().cuda().to(memory_format=torch.channels_last)
optimizer = torch.optim.SGD(model.parameters(), .03, momentum=.9)
beginning = time.perf_counter()
accuracy = 0
for step in range(500):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()
    optimizer.step()
    if (step + 1) % 25 == 0:
        model.eval()
        with torch.no_grad():
            accuracy = (model(x).argmax(1) == y).float().mean().item() * 100
        if accuracy >= 99:
            break
result = {"samples": 256, "augmentation": False, "steps": step + 1, "accuracy": accuracy,
          "seconds": time.perf_counter() - beginning, "passed": accuracy >= 95,
          "purpose": "implementation validation only; not a formal test result"}
dump_json("validation/overfit.json", result)
print(json.dumps(result))
assert result["passed"]
