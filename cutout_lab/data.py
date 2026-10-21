import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

from .utils import dump_json, sha256

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def stratified_split(labels, seed=2026, validation_per_class=500):
    rng = np.random.default_rng(seed)
    training, validation, probe = [], [], []
    for category in range(10):
        ids = rng.permutation(np.flatnonzero(np.asarray(labels) == category))
        validation.extend(ids[:validation_per_class].tolist())
        remaining = ids[validation_per_class:]
        training.extend(remaining.tolist())
        probe.extend(remaining[:100].tolist())
    return {"seed": seed, "train": sorted(training), "validation": sorted(validation), "probe": sorted(probe)}


def prepare(root="data", download=False):
    root = Path(root)
    source = CIFAR10(str(root), train=True, download=download)
    test = CIFAR10(str(root), train=False, download=download)
    path = root / "split.json"
    expected = stratified_split(source.targets)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Existing split differs from the frozen stratified split")
    else:
        dump_json(path, expected)
    stats_path = root / "normalization.json"
    if not stats_path.exists():
        # Accumulate sums in float64 without constructing a large float64 image array.
        sums, squares = np.zeros(3), np.zeros(3)
        ids = expected["train"]
        for start in range(0, len(ids), 1000):
            x = source.data[ids[start:start + 1000]].astype(np.float64) / 255.0
            sums += x.sum((0, 1, 2))
            squares += (x * x).sum((0, 1, 2))
        count = len(ids) * 32 * 32
        mean = sums / count
        std = np.sqrt(squares / count - mean * mean)
        dump_json(stats_path, {"mean": mean.tolist(), "std": std.tolist(),
                              "source": "45000 training images only", "split_sha256": sha256(path)})
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats["split_sha256"] != sha256(path):
        raise ValueError("Normalization does not correspond to the split")
    return source, test, expected, stats


class ResidentData:
    """Keep this small dataset on the device, avoiding Windows worker overhead."""
    def __init__(self, root="data", device="cuda", download=False):
        source, test, split, stats = prepare(root, download)
        self.device = torch.device(device)
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)[None, :, None, None]
        self.std = torch.tensor(stats["std"], dtype=torch.float32, device=device)[None, :, None, None]
        self.stats, self.split = stats, split
        self.split_hash = sha256(Path(root) / "split.json")
        self.source, self.test_source = source, test
        raw = torch.from_numpy(source.data[split["train"]].transpose(0, 3, 1, 2).copy()).to(device)
        self.padded = (F.pad(raw.float() / 255., (4, 4, 4, 4)) - self.mean) / self.std
        self.labels = torch.tensor(np.asarray(source.targets)[split["train"]], device=device)
        self.validation = self.normalized(source.data[split["validation"]])
        self.validation_labels = torch.tensor(np.asarray(source.targets)[split["validation"]], device=device)
        self.probe = self.normalized(source.data[split["probe"]])
        self.probe_labels = torch.tensor(np.asarray(source.targets)[split["probe"]], device=device)
        self.test = None
        self.test_labels = None

    def normalized(self, array):
        x = torch.from_numpy(array.transpose(0, 3, 1, 2).copy()).to(self.device).float() / 255.
        return ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)

    def load_test(self):
        if self.test is None:
            self.test = self.normalized(self.test_source.data)
            self.test_labels = torch.tensor(self.test_source.targets, device=self.device)


def probability(method, epoch, epochs):
    if method == "baseline":
        return 0.0
    if method == "cutout":
        return 1.0
    if method == "matched":
        return 0.5
    if method == "progressive":
        if epochs < 2:
            raise ValueError("Progressive schedule needs at least two epochs")
        return epoch / (epochs - 1)
    raise ValueError(method)


def epoch_plan(seed, epoch, count):
    # The independent streams guarantee matched minibatches/base augmentation across methods.
    base = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0]))
    mask = np.random.default_rng(np.random.SeedSequence([seed, epoch, 1]))
    return {"order": base.permutation(count), "dy": base.integers(0, 9, count),
            "dx": base.integers(0, 9, count), "flip": base.random(count) < .5,
            "u": mask.random(count), "cy": mask.integers(0, 32, count),
            "cx": mask.integers(0, 32, count)}


def make_mask(cy, cx, active, length=16, size=32):
    axis = torch.arange(size, device=cy.device)
    square = ((axis[None, :, None] >= cy[:, None, None] - length // 2)
              & (axis[None, :, None] < cy[:, None, None] + (length + 1) // 2)
              & (axis[None, None, :] >= cx[:, None, None] - length // 2)
              & (axis[None, None, :] < cx[:, None, None] + (length + 1) // 2))
    return square & active[:, None, None]


def augmented_batch(padded, ids, plan, probability_value, length=16):
    coords = torch.arange(32, device=padded.device)
    channels = torch.arange(3, device=padded.device)
    dy, dx = plan["dy"][ids], plan["dx"][ids]
    yy = coords[None, :] + dy[:, None]
    xx = coords[None, :] + dx[:, None]
    xx = torch.where(plan["flip"][ids, None], xx.flip(1), xx)
    x = padded[ids[:, None, None, None], channels[None, :, None, None],
               yy[:, None, :, None], xx[:, None, None, :]]
    mask = make_mask(plan["cy"][ids], plan["cx"][ids], plan["u"][ids] < probability_value, length)
    x = x.masked_fill(mask[:, None], 0)
    return x.contiguous(memory_format=torch.channels_last), mask


def test_positions(count, length, repeat):
    rng = np.random.default_rng(np.random.SeedSequence([2026, length, repeat, 991]))
    return rng.integers(0, 33 - length, size=(count, 2), dtype=np.int64)


def apply_test_occlusion(x, positions, length):
    axis = torch.arange(32, device=x.device)
    y, z = positions[:, 0], positions[:, 1]
    mask = ((axis[None, :, None] >= y[:, None, None])
            & (axis[None, :, None] < y[:, None, None] + length)
            & (axis[None, None, :] >= z[:, None, None])
            & (axis[None, None, :] < z[:, None, None] + length))
    return x.masked_fill(mask[:, None], 0).contiguous(memory_format=torch.channels_last)
