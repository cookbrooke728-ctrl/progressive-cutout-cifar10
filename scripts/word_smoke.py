"""Internal Word rendering fixture, no experiment results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
from build_paper import Document, configure_styles, add_text, add_table

doc = Document()
configure_styles(doc)
add_text(doc, "论文排版渲染检查", 17, style="Title", center=True)
add_text(doc, "刘凌霄　华北电力大学", 10.5, center=True)
add_text(doc, "Lingxiao Liu　North China Electric Power University", 10, center=True)
doc.add_paragraph("方法与环境", style="Heading 1")
add_text(doc, "此文件仅用于检查本机Word对中文、英文、表格和页码的渲染能力，不包含任何正式实验性能结果。正式论文由完整训练与评估记录生成。")
add_table(doc, "排版检查表", ["项目", "内容"], [["模型", "CIFAR ResNet-20"], ["参数量", "269722"], ["真实署名", "刘凌霄"]], [4, 11.7])
Path(".tmp").mkdir(exist_ok=True)
doc.save(".tmp/render_smoke.docx")
