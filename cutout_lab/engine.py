import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import (CLASSES, apply_test_occlusion, augmented_batch, epoch_plan,
                   probability, test_positions)
from .model import ResNet20, parameter_count
from .utils import dump_json, metrics, provenance, seed_everything, sha256, write_csv

METHODS = ("baseline", "cutout", "matched", "progressive")


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict(model, x, labels, batch_size=512, positions=None, length=0):
    model.eval()
    losses = torch.zeros((), device=x.device)
    predictions, confidences = [], []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size]
        if positions is not None:
            xb = apply_test_occlusion(xb, positions[start:start + batch_size], length)
        output = model(xb)
        losses += F.cross_entropy(output, labels[start:start + batch_size], reduction="sum")
        confidence, predicted = output.softmax(1).max(1)
        predictions.append(predicted)
        confidences.append(confidence)
    predicted = torch.cat(predictions).cpu().numpy()
    confidence = torch.cat(confidences).cpu().numpy()
    result = metrics(labels.cpu().numpy(), predicted)
    result["loss"] = float(losses.item() / len(x))
    return result, predicted, confidence


def train_epoch(model, optimizer, data, method, seed, epoch, epochs, batch_size=128):
    model.train()
    plan = {k: torch.as_tensor(v, device=data.device) for k, v in epoch_plan(seed, epoch, len(data.labels)).items()}
    prob = probability(method, epoch, epochs)
    running_loss = torch.zeros((), device=data.device)
    correct = torch.zeros((), dtype=torch.int64, device=data.device)
    covered = torch.zeros((), dtype=torch.int64, device=data.device)
    selected = torch.zeros((), dtype=torch.int64, device=data.device)
    for start in range(0, len(data.labels), batch_size):
        ids = plan["order"][start:start + batch_size]
        xb, mask = augmented_batch(data.padded, ids, plan, prob)
        target = data.labels[ids]
        if hasattr(model, "_orig_mod"):
            torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        output = model(xb)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach() * len(ids)
        correct += (output.detach().argmax(1) == target).sum()
        covered += mask.sum()
        selected += (plan["u"][ids] < prob).sum()
    count = len(data.labels)
    return {"train_loss": running_loss.item() / count, "train_aug_accuracy": correct.item() / count * 100,
            "p": prob, "masked_images": selected.item(), "masked_pixel_fraction": covered.item() / (count * 1024)}


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train_run(data, method, seed, epochs=100, batch_size=128, output="runs", resume=False,
              stop_after=None, run_dir=None):
    if epochs < 2 or method not in METHODS:
        raise ValueError("Need a supported method and at least 2 epochs")
    run = Path(run_dir) if run_dir else Path(output) / f"{method}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    config = {"method": method, "seed": seed, "epochs": epochs, "batch_size": batch_size,
              "architecture": "cifar_resnet20_option_a", "lr": .1, "min_lr": .0001,
              "momentum": .9, "weight_decay": .0005, "length": 16,
              "split_sha256": data.split_hash, "normalization": data.stats,
              "classes": CLASSES, "precision": "float32; TF32 disabled", "deterministic_cudnn": True,
              "execution_backend": "cudagraphs" if data.device.type == "cuda" else "eager",
              "sgd_fused": data.device.type == "cuda"}
    seed_everything(seed)
    model = ResNet20().to(data.device, memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.0005,
                                fused=data.device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.0001)
    history, start_epoch, best_accuracy, best_epoch = [], 0, -1., None
    if resume:
        checkpoint = torch.load(run / "last.pt", map_location=data.device, weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("Resume configuration differs from frozen checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        start_epoch, best_accuracy, best_epoch = checkpoint["next_epoch"], checkpoint["best_accuracy"], checkpoint["best_epoch"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if data.device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
    else:
        if (run / "last.pt").exists():
            raise FileExistsError(f"{run} already contains a run; use --resume")
        dump_json(run / "config.json", config)
        dump_json(run / "provenance.json", {**provenance(), "started_utc": datetime.now(timezone.utc).isoformat(),
                                           "parameters": parameter_count()})
        # Record a hash of identical initialization without retaining another weight file.
        import hashlib
        h = hashlib.sha256()
        for key, tensor in model.state_dict().items():
            h.update(key.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
        dump_json(run / "initialization.json", {"sha256": h.hexdigest(), "seed": seed})
    training_model = torch.compile(model, backend="cudagraphs", fullgraph=True, dynamic=False) if data.device.type == "cuda" else model
    end_epoch = min(epochs, start_epoch + stop_after) if stop_after else epochs
    for epoch in range(start_epoch, end_epoch):
        sync(data.device)
        beginning = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        row = train_epoch(training_model, optimizer, data, method, seed, epoch, epochs, batch_size)
        validation, _, _ = predict(model, data.validation, data.validation_labels)
        probe, _, _ = predict(model, data.probe, data.probe_labels)
        scheduler.step()
        sync(data.device)
        row = {"epoch": epoch + 1, "lr": lr, **row, "validation_loss": validation["loss"],
               "validation_accuracy": validation["accuracy"], "probe_loss": probe["loss"],
               "probe_accuracy": probe["accuracy"], "seconds": time.perf_counter() - beginning}
        history.append(row)
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch + 1
            atomic_torch_save({"state_dict": cpu_state(model), "config": config,
                               "best_epoch": best_epoch, "validation_accuracy": best_accuracy}, run / "best.pt")
        atomic_torch_save({"model": cpu_state(model), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "config": config, "history": history,
                           "next_epoch": epoch + 1, "best_accuracy": best_accuracy, "best_epoch": best_epoch,
                           "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                           "torch_rng": torch.get_rng_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if data.device.type == "cuda" else []}, run / "last.pt")
        write_csv(run / "history.csv", history)
        dump_json(run / "status.json", {"method": method, "seed": seed, "completed_epochs": epoch + 1,
                                       "total_epochs": epochs, "best_epoch": best_epoch,
                                       "best_validation_accuracy": best_accuracy, "complete": epoch + 1 == epochs,
                                       "training_seconds": sum(r["seconds"] for r in history),
                                       "best_weight_sha256": sha256(run / "best.pt")})
        print(json.dumps({"run": run.name, **row, "best_validation": best_accuracy}), flush=True)
    if data.device.type == "cuda":
        del training_model
        torch._dynamo.reset()
        __import__("gc").collect()
        torch.cuda.empty_cache()
    return model, history


def evaluate_run(data, run, force=False):
    run = Path(run)
    from .weights import resolve_weight
    weight = resolve_weight(run)
    if (run / "evaluation.json").exists() and not force:
        result = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
        if result["weight_sha256"] != sha256(weight):
            raise ValueError("Weights changed after evaluation")
        return result
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    if not status["complete"]:
        raise ValueError("Formal test evaluation requires completed training")
    checkpoint = torch.load(weight, map_location=data.device, weights_only=True)
    if checkpoint["config"]["split_sha256"] != data.split_hash or checkpoint["config"]["normalization"] != data.stats:
        raise ValueError("Evaluation preprocessing/split differs from training")
    data.load_test()
    model = ResNet20().to(data.device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["state_dict"])
    labels = data.test_labels.cpu().numpy()
    beginning = time.perf_counter()
    result = {"method": checkpoint["config"]["method"], "seed": checkpoint["config"]["seed"],
              "best_epoch": checkpoint["best_epoch"], "validation_accuracy": checkpoint["validation_accuracy"],
              "weight_sha256": sha256(weight), "split_sha256": data.split_hash,
              "test_samples": len(labels), "occlusion": {}}
    def save_predictions(name, predictions, confidences):
        write_csv(run / "predictions" / f"{name}.csv", ({"index": i, "target": int(y),
                  "prediction": int(p), "confidence": float(c)}
                 for i, (y, p, c) in enumerate(zip(labels, predictions, confidences))))
    clean, predicted, confidence = predict(model, data.test, data.test_labels)
    result["clean"] = clean
    save_predictions("clean", predicted, confidence)
    for length in (8, 16, 24):
        repeated = []
        for repeat in range(3):
            pos = test_positions(len(labels), length, repeat)
            digest = __import__("hashlib").sha256(pos.tobytes()).hexdigest()
            pos = torch.tensor(pos, device=data.device)
            score, predicted, confidence = predict(model, data.test, data.test_labels, positions=pos, length=length)
            score["positions_sha256"] = digest
            repeated.append(score)
            save_predictions(f"occlusion_{length}_repeat{repeat}", predicted, confidence)
        result["occlusion"][str(length)] = {"accuracy": float(np.mean([v["accuracy"] for v in repeated])),
                                            "repeats": repeated}
    result["mean_occluded_accuracy"] = float(np.mean([v["accuracy"] for v in result["occlusion"].values()]))
    result["evaluation_seconds"] = time.perf_counter() - beginning
    dump_json(run / "evaluation.json", result)
    print(json.dumps({"evaluated": run.name, "clean": clean["accuracy"], "occluded": result["mean_occluded_accuracy"]}), flush=True)
    return result
