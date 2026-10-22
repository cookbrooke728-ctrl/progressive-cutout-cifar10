"""Package the audited, published and download-verified course deliverables."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
audit = json.loads((root / "output/audit.json").read_text(encoding="utf-8"))
verification = json.loads((root / "output/download_verification.json").read_text(encoding="utf-8"))
assert audit["complete_runs"] == 12 and verification["all_passed"]
assert len(verification["weights"]) == 12
source = root / ".tmp/submission-source"
subprocess.run([sys.executable, str(root / "scripts/prepare_public_source.py"),
                "--destination", ".tmp/submission-source"], cwd=root, check=True)
files = {p.relative_to(source).as_posix(): p for p in source.rglob("*") if p.is_file()}
manifest = json.loads((root / "release-assets/manifest.json").read_text(encoding="utf-8"))
for entry in manifest["weights"]:
    path = root / "release-assets" / entry["filename"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    files["weights/" + path.name] = path
files["weights/manifest.json"] = root / "release-assets/manifest.json"
summary = json.loads((root / "output/summary.json").read_text(encoding="utf-8"))
publication = json.loads((root / "paper/publication.json").read_text(encoding="utf-8"))
readme = ["# 课程论文提交包", "", "刘凌霄 · 华北电力大学", "",
          "论文：output/paper/progressive_cutout_paper.docx 与同名 PDF。", "",
          "代码、配置、原始日志、逐样本预测、图表和十二份最佳权重均包含在本包中。",
          "运行方法见 README.md；权重位于 weights/，每份文件可用 SHA256SUMS.txt 核对。", "",
          f"公开代码：{publication['repository_url']}", f"公开权重与记录：{publication['release_url']}", "",
          "## 实测结果", "", "以下为三个训练种子的均值 ± 样本标准差，单位为百分比。", ""]
for name, result in summary["methods"].items():
    a, o = result["accuracy"], result["mean_occluded_accuracy"]
    readme.append(f"- {name}：干净准确率 {a['mean']:.2f} ± {a['std']:.2f}；遮挡平均准确率 {o['mean']:.2f} ± {o['std']:.2f}。")
readme.extend(["", "全部十二份 GitHub Release 权重已重新下载，哈希与完整干净及遮挡评估均通过。",
               "核验详情见 output/download_verification.json。", "",
               "原始 CIFAR-10 图像和本地 Python 环境需按 README 单独准备。"])
destination = root / "output/submission_bundle.zip"
checksums = []
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    def add(name, content):
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, content)
        checksums.append(f"{hashlib.sha256(content).hexdigest()}  {name}")
    for name, path in sorted(files.items()):
        add(name, path.read_bytes())
    add("提交说明.md", ("\n".join(readme) + "\n").encode("utf-8"))
    archive.writestr("SHA256SUMS.txt", "\n".join(checksums) + "\n")
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
print(json.dumps({"bundle": str(destination), "files": len(files) + 2,
                  "bytes": destination.stat().st_size}, ensure_ascii=False))
