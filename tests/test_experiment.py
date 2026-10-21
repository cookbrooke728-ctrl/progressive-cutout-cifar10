import json

import numpy as np
import pytest
import torch

from cutout_lab.data import (apply_test_occlusion, augmented_batch, epoch_plan, make_mask,
                             probability, stratified_split, test_positions as occlusion_positions)
from cutout_lab.engine import train_run
from cutout_lab.model import ResNet20, parameter_count
from cutout_lab.utils import metrics


def test_architecture():
    torch.set_num_threads(1)
    m = ResNet20()
    assert parameter_count() == 269722
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_split_no_leakage():
    labels = np.repeat(np.arange(10), 5000)
    split = stratified_split(labels)
    assert len(split["train"]) == 45000 and len(split["validation"]) == 5000
    assert not set(split["train"]) & set(split["validation"])
    assert set(split["train"]) | set(split["validation"]) == set(range(50000))
    assert set(split["probe"]) <= set(split["train"])
    assert np.bincount(labels[split["validation"]]).tolist() == [500] * 10


def test_schedule_and_mask_edges():
    assert probability("progressive", 0, 100) == 0
    assert probability("progressive", 99, 100) == 1
    assert np.mean([probability("progressive", e, 100) for e in range(100)]) == pytest.approx(.5)
    mask = make_mask(torch.tensor([0, 16, 31]), torch.tensor([0, 16, 31]), torch.tensor([True, True, False]))
    assert mask.sum((1, 2)).tolist() == [64, 256, 0]


def test_paired_augmentation_and_probability_zero():
    padded = torch.arange(8 * 3 * 40 * 40).reshape(8, 3, 40, 40).float() + 1
    plan = {k: torch.as_tensor(v) for k, v in epoch_plan(42, 0, 8).items()}
    ids = plan["order"]
    base, no_mask = augmented_batch(padded, ids, plan, 0)
    augmented, mask = augmented_batch(padded, ids, plan, 1)
    assert no_mask.sum() == 0
    assert torch.equal(base.masked_fill(mask[:, None], 0), augmented)
    for i, original in enumerate(ids):
        dy, dx = int(plan["dy"][original]), int(plan["dx"][original])
        crop = padded[original, :, dy:dy + 32, dx:dx + 32]
        if plan["flip"][original]:
            crop = crop.flip(-1)
        assert torch.equal(crop, base[i])


def test_test_occlusion_is_shared_and_exact_area():
    positions = occlusion_positions(100, 16, 0)
    assert np.array_equal(positions, occlusion_positions(100, 16, 0))
    x = apply_test_occlusion(torch.ones(100, 3, 32, 32), torch.tensor(positions), 16)
    assert ((x == 0).sum((1, 2, 3)) == 3 * 256).all()


def test_macro_f1():
    result = metrics(np.arange(10), np.arange(10))
    assert result["accuracy"] == result["macro_f1"] == 100


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_backend_preserves_one_step_sgd():
    import copy
    from cutout_lab.utils import seed_everything
    seed_everything(123)
    reference = ResNet20().cuda().to(memory_format=torch.channels_last)
    candidate = copy.deepcopy(reference)
    x = torch.randn(8, 3, 32, 32, device="cuda").to(memory_format=torch.channels_last)
    y = torch.arange(8, device="cuda")
    compiled = torch.compile(candidate, backend="cudagraphs", fullgraph=True, dynamic=False)
    for model, fused in ((reference, False), (compiled, True)):
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0005, fused=fused)
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
    torch.cuda.synchronize()
    for key, value in reference.state_dict().items():
        assert torch.allclose(value, candidate.state_dict()[key], atol=2e-5, rtol=2e-4), key
    assert candidate.bn.num_batches_tracked.item() == 1
    torch._dynamo.reset()


class TinyData:
    def __init__(self):
        generator = torch.Generator().manual_seed(5)
        self.device = torch.device("cpu")
        self.padded = torch.randn(16, 3, 40, 40, generator=generator)
        self.labels = torch.arange(16) % 10
        self.validation = torch.randn(10, 3, 32, 32, generator=generator)
        self.validation_labels = torch.arange(10)
        self.probe, self.probe_labels = self.validation, self.validation_labels
        self.split_hash, self.stats = "synthetic-test-only", {"mean": [0, 0, 0], "std": [1, 1, 1]}


def test_resume_matches_continuous_training(tmp_path):
    data = TinyData()
    direct, h1 = train_run(data, "progressive", 42, 3, 8, run_dir=tmp_path / "direct")
    train_run(data, "progressive", 42, 3, 8, stop_after=1, run_dir=tmp_path / "resumed")
    resumed, h2 = train_run(data, "progressive", 42, 3, 8, resume=True, run_dir=tmp_path / "resumed")
    for key, tensor in direct.state_dict().items():
        assert torch.equal(tensor, resumed.state_dict()[key]), key
    for a, b in zip(h1, h2):
        for key in a:
            if key != "seconds":
                assert a[key] == b[key]
    assert len(h2) == 3 and [r["p"] for r in h2] == [0, .5, 1]
    with pytest.raises(ValueError, match="configuration"):
        train_run(data, "progressive", 42, 4, 8, resume=True, run_dir=tmp_path / "resumed")
