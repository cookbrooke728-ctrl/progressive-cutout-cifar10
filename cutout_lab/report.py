import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from .data import CLASSES, prepare, test_positions
from .engine import METHODS
from .utils import dump_json, sha256, write_csv

NAMES = {"baseline": "Baseline", "cutout": "Cutout p=1", "matched": "Cutout p=0.5", "progressive": "Progressive Cutout"}
COLORS = {"baseline": "#6B7280", "cutout": "#D97706", "matched": "#059669", "progressive": "#2563EB"}


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build_report(runs="runs", output="output"):
    runs, output = Path(runs), Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    records, histories, configs = {}, {}, []
    audit = []
    shared_targets = None
    for method in METHODS:
        for seed in (42, 43, 44):
            run = runs / f"{method}_seed{seed}"
            evaluation = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            status = json.loads((run / "status.json").read_text(encoding="utf-8"))
            assert status["complete"] and status["completed_epochs"] == config["epochs"]
            assert evaluation["weight_sha256"] == sha256(run / "best.pt")
            assert config["method"] == method and config["seed"] == seed
            records[method, seed] = evaluation
            histories[method, seed] = read_csv(run / "history.csv")
            assert [int(r["epoch"]) for r in histories[method, seed]] == list(range(1, config["epochs"] + 1))
            configs.append(config)
            audit.append({"run": run.name, "weight_sha256": evaluation["weight_sha256"],
                          "history_sha256": sha256(run / "history.csv"),
                          "evaluation_sha256": sha256(run / "evaluation.json"),
                          "clean_predictions_sha256": sha256(run / "predictions" / "clean.csv")})
            # Recompute the primary number directly from saved predictions.
            predictions = read_csv(run / "predictions" / "clean.csv")
            assert [int(r["index"]) for r in predictions] == list(range(10000))
            targets = [int(r["target"]) for r in predictions]
            if shared_targets is None:
                shared_targets = targets
            assert targets == shared_targets
            accuracy = np.mean([r["target"] == r["prediction"] for r in predictions]) * 100
            assert len(predictions) == 10000 and abs(accuracy - evaluation["clean"]["accuracy"]) < 1e-10
            from .utils import metrics
            recalculated = metrics([int(r["target"]) for r in predictions], [int(r["prediction"]) for r in predictions])
            assert abs(recalculated["macro_f1"] - evaluation["clean"]["macro_f1"]) < 1e-10
            assert recalculated["confusion"] == evaluation["clean"]["confusion"]
            audit[-1]["occluded_predictions_sha256"] = {}
            for length in (8, 16, 24):
                for repeat in range(3):
                    occluded = read_csv(run / "predictions" / f"occlusion_{length}_repeat{repeat}.csv")
                    assert len(occluded) == 10000
                    assert [int(r["index"]) for r in occluded] == list(range(10000))
                    assert [int(r["target"]) for r in occluded] == shared_targets
                    digest = __import__("hashlib").sha256(test_positions(10000, length, repeat).tobytes()).hexdigest()
                    assert digest == evaluation["occlusion"][str(length)]["repeats"][repeat]["positions_sha256"]
                    audit[-1]["occluded_predictions_sha256"][f"{length}_{repeat}"] = sha256(run / "predictions" / f"occlusion_{length}_repeat{repeat}.csv")
                    oa = np.mean([r["target"] == r["prediction"] for r in occluded]) * 100
                    assert abs(oa - evaluation["occlusion"][str(length)]["repeats"][repeat]["accuracy"]) < 1e-10
    common = [{k: v for k, v in c.items() if k not in ("method", "seed")} for c in configs]
    assert all(c == common[0] for c in common)
    for seed in (42, 43, 44):
        initial = [json.loads((runs / f"{method}_seed{seed}" / "initialization.json").read_text())["sha256"] for method in METHODS]
        assert len(set(initial)) == 1
    source_hashes = [json.loads((runs / f"{method}_seed{seed}" / "provenance.json").read_text())["training_source_sha256"] for method in METHODS for seed in (42, 43, 44)]
    assert all(hashes == source_hashes[0] for hashes in source_hashes)
    def array(method, key):
        return np.array([records[method, seed]["clean"][key] if key in ("accuracy", "macro_f1")
                         else records[method, seed][key] for seed in (42, 43, 44)])
    summary = {"epochs": configs[0]["epochs"], "configuration": common[0], "methods": {}, "paired": {}, "runs": []}
    for method in METHODS:
        result = {}
        for key in ("accuracy", "macro_f1", "mean_occluded_accuracy"):
            values = array(method, key)
            result[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "values": values.tolist()}
        for length in (8, 16, 24):
            values = np.array([records[method, s]["occlusion"][str(length)]["accuracy"] for s in (42, 43, 44)])
            result[f"occlusion_{length}"] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
        result["training_hours"] = sum(sum(float(r["seconds"]) for r in histories[method, s]) for s in (42, 43, 44)) / 3600
        result["best_epochs"] = [records[method, s]["best_epoch"] for s in (42, 43, 44)]
        result["per_class"] = np.mean([records[method, s]["clean"]["per_class_accuracy"] for s in (42, 43, 44)], axis=0).tolist()
        result["probe_val_gap_final"] = float(np.mean([float(histories[method, s][-1]["probe_accuracy"]) - float(histories[method, s][-1]["validation_accuracy"]) for s in (42, 43, 44)]))
        early = np.array([np.mean([float(r["validation_accuracy"]) for r in histories[method, s][:max(1, configs[0]["epochs"] // 5)]]) for s in (42, 43, 44)])
        result["early_validation_mean"] = {"mean": float(early.mean()), "std": float(early.std(ddof=1)), "window_epochs": max(1, configs[0]["epochs"] // 5)}
        result["realized_mask_fraction"] = float(np.mean([float(r["masked_pixel_fraction"]) for s in (42, 43, 44) for r in histories[method, s]]))
        summary["methods"][method] = result
        for seed in (42, 43, 44):
            e = records[method, seed]
            summary["runs"].append({"method": method, "seed": seed, "best_epoch": e["best_epoch"],
                                    "accuracy": e["clean"]["accuracy"], "macro_f1": e["clean"]["macro_f1"],
                                    "mean_occluded_accuracy": e["mean_occluded_accuracy"], "weight_sha256": e["weight_sha256"]})
    for left, right in (("cutout", "baseline"), ("progressive", "matched"), ("progressive", "cutout")):
        paired = {}
        for key in ("accuracy", "mean_occluded_accuracy"):
            delta = array(left, key) - array(right, key)
            paired[key] = {"mean": float(delta.mean()), "std": float(delta.std(ddof=1)),
                           "values": delta.tolist(), "positive_seeds": int((delta > 0).sum())}
        summary["paired"][f"{left}_minus_{right}"] = paired
    write_csv(output / "tables" / "all_runs.csv", summary["runs"])
    write_csv(output / "tables" / "main_results.csv", ({"method": m,
              "clean_mean": v["accuracy"]["mean"], "clean_sd": v["accuracy"]["std"],
              "macro_f1_mean": v["macro_f1"]["mean"], "macro_f1_sd": v["macro_f1"]["std"],
              "occluded_mean": v["mean_occluded_accuracy"]["mean"], "occluded_sd": v["mean_occluded_accuracy"]["std"],
              "training_hours_3_runs": v["training_hours"]} for m, v in summary["methods"].items()))
    write_csv(output / "tables" / "per_class.csv", ({"class": c, **{m: summary["methods"][m]["per_class"][i] for m in METHODS}} for i, c in enumerate(CLASSES)))
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "font.family": "DejaVu Sans"})
    def save(fig, name):
        fig.savefig(figures / f"{name}.png", bbox_inches="tight")
        fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
    # Fig. 1: workflow and the genuinely implemented schedule.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.set_axis_off()
    for y, label in ((.83, "Fixed train / validation split"), (.59, "Paired crop and flip"), (.35, "Method-specific occlusion"), (.11, "ResNet-20 + cross-entropy")):
        ax.add_patch(FancyBboxPatch((.08, y - .07), .82, .15, boxstyle="round,pad=.02", facecolor="#F1F5F9", edgecolor="#94A3B8"))
        ax.text(.49, y, label, ha="center", va="center", fontsize=12)
        if y > .2:
            ax.annotate("", (.49, y - .16), (.49, y - .09), arrowprops={"arrowstyle": "->"})
    e = np.arange(configs[0]["epochs"])
    for m, p in (("baseline", np.zeros(len(e))), ("cutout", np.ones(len(e))), ("matched", np.full(len(e), .5)), ("progressive", e / (len(e) - 1))):
        axes[1].plot(e + 1, p, label=NAMES[m], color=COLORS[m], linewidth=2)
    axes[1].set(xlabel="Epoch", ylabel="Occlusion probability", ylim=(-.05, 1.05))
    axes[1].legend(fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig1_method")
    source, test, _, stats = prepare("data")
    mean = np.array(stats["mean"])
    fig, axes = plt.subplots(2, 5, figsize=(9, 3.8))
    for i in range(5):
        index = int(np.flatnonzero(np.array(test.targets) == i)[0])
        original = test.data[index] / 255.
        masked = original.copy()
        masked[8:24, 8:24] = mean
        axes[0, i].imshow(original)
        axes[1, i].imshow(masked)
        axes[0, i].set_title(CLASSES[i], fontsize=14)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    fig.suptitle("Original (top); 16 x 16 mean-filled occlusion (bottom)", fontsize=14)
    fig.tight_layout()
    save(fig, "fig2_examples")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for method in METHODS:
        for axis, field, label in zip(axes, ("train_loss", "validation_accuracy", "probe_accuracy"), ("Training loss", "Validation accuracy (%)", "Probe accuracy (%)")):
            values = np.array([[float(r[field]) for r in histories[method, s]] for s in (42, 43, 44)])
            avg, sd = values.mean(0), values.std(0, ddof=1)
            axis.plot(e + 1, avg, color=COLORS[method], label=NAMES[method], linewidth=1.4)
            axis.fill_between(e + 1, avg - sd, avg + sd, color=COLORS[method], alpha=.13)
            axis.set(xlabel="Epoch", ylabel=label)
            axis.xaxis.label.set_size(13)
            axis.yaxis.label.set_size(13)
            axis.grid(alpha=.2)
    axes[0].legend(fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig3_learning")
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    for method in METHODS:
        values = np.array([[records[method, s]["clean"]["accuracy"], *[records[method, s]["occlusion"][str(l)]["accuracy"] for l in (8, 16, 24)]] for s in (42, 43, 44)])
        ax.errorbar([0, 8, 16, 24], values.mean(0), yerr=values.std(0, ddof=1), label=NAMES[method], color=COLORS[method], marker="o", capsize=3)
    ax.set(xlabel="Test occlusion side length (pixels)", ylabel="Accuracy (%)", xticks=[0, 8, 16, 24])
    ax.legend(fontsize=11)
    ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, "fig4_robustness")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    for method, ax in zip(("baseline", "progressive"), axes):
        cm = np.mean([records[method, s]["clean"]["confusion"] for s in (42, 43, 44)], axis=0) / 10
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        ax.set(title=NAMES[method], xticks=range(10), yticks=range(10), xlabel="Predicted class", ylabel="True class")
        for i in range(10):
            for j in range(10):
                if cm[i, j] >= 1:
                    ax.text(j, i, f"{cm[i,j]:.0f}", ha="center", va="center", fontsize=10, color="white" if cm[i,j] > 60 else "black")
    fig.colorbar(im, ax=axes, label="Row percentage", shrink=.8)
    save(fig, "fig5_confusion")
    b = read_csv(runs / "baseline_seed42" / "predictions" / "clean.csv")
    p = read_csv(runs / "progressive_seed42" / "predictions" / "clean.csv")
    improvements = [i for i in range(10000) if b[i]["prediction"] != b[i]["target"] and p[i]["prediction"] == p[i]["target"]]
    regressions = [i for i in range(10000) if b[i]["prediction"] == b[i]["target"] and p[i]["prediction"] != p[i]["target"]]
    summary["cases"] = {"rule": "seed42 clean test; ascending original index; first three in each direction", "improved_count": len(improvements), "regressed_count": len(regressions), "improved_indices": improvements[:3], "regressed_indices": regressions[:3]}
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for row, selected in enumerate((improvements[:3], regressions[:3])):
        for col in range(3):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(selected):
                ax.text(.5, .5, "No further cases", ha="center")
                continue
            idx = selected[col]
            ax.imshow(test.data[idx])
            title = f"#{idx} True: {CLASSES[test.targets[idx]]}\nB: {CLASSES[int(b[idx]['prediction'])]} / P: {CLASSES[int(p[idx]['prediction'])]}"
            ax.set_title(title, fontsize=12)
    fig.suptitle("First three improvements (top) and regressions (bottom), seed 42", fontsize=12)
    fig.tight_layout()
    save(fig, "fig6_cases")
    dump_json(output / "summary.json", summary)
    dump_json(output / "audit.json", {"complete_runs": 12, "paired_initializations": True, "training_source_consistent": True, "prediction_metrics_verified": True, "artifacts": audit})
    print(json.dumps(summary["methods"], indent=2))
