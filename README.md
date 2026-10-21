# Progressive Cutout on CIFAR-10

课程论文配套实验：复现 Cutout，在 CIFAR ResNet-20 上比较固定概率和线性渐进概率遮挡。
所有正式实验均从零训练；不使用公开预训练权重代替本文训练结果。

## Environment

Windows / Linux, Python 3.10+ (本机实际为 3.13.5), NVIDIA GPU。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m cutout_lab prepare
```

训练环境与文档生成环境相互独立。`requirements.txt` 用于训练和分析。

## Frozen experiment

官方 CIFAR-10 训练集按类别固定划分为 45,000 / 5,000 张，测试集完整保留 10,000 张。
划分索引见 `data/split.json`，归一化只使用 45,000 张训练图像，统计见 `data/normalization.json`。
模型为 option-A CIFAR ResNet-20，269,722 个参数，无预训练。

| Method | Probability | Length | Purpose |
|---|---:|---:|---|
| baseline | 0 | 16 | Crop/flip baseline |
| cutout | 1 | 16 | Original Cutout strategy |
| matched | 0.5 | 16 | Equal expected exposure control |
| progressive | epoch / (epochs - 1), zero-based epoch | 16 | Linear probability schedule |

训练遮挡中心允许在图像边缘，超出部分截断；归一化后填零。
基础增强为原像素零填充 4 像素后随机裁剪至 32×32、概率 0.5 水平翻转。
同种子方法间共享初始化与基础增强，遮挡随机数独立。
SGD, LR=0.1, momentum=0.9, weight decay=0.0005, batch=128；每轮余弦衰减至 0.0001，无重启。
参数和输入为 FP32，统一关闭 TF32 与 cuDNN benchmark，启用确定性卷积。
正式GPU训练统一使用PyTorch `cudagraphs`执行后端及融合SGD，减少Windows内核提交开销；无额外Triton依赖。
不同执行后端的浮点舍入可导致训练轨迹差异，跨硬件或后端不承诺逐位相同。
按干净验证准确率选最佳检查点，同分取更早轮次。完整配置见 `configs/experiment.json`。

```powershell
# Entire suite: resume automatically; skip complete runs
.\.venv\Scripts\python.exe -m cutout_lab suite
# Limit a session to four completed runs (one seed across four methods)
.\.venv\Scripts\python.exe -m cutout_lab suite --stop-after-runs 4
# Single run; --stop-after saves a session checkpoint without changing total schedule
.\.venv\Scripts\python.exe -m cutout_lab train --method progressive --seed 42 --epochs 100 --stop-after 10
.\.venv\Scripts\python.exe -m cutout_lab train --method progressive --seed 42 --epochs 100 --resume
# Only after frozen training is complete
.\.venv\Scripts\python.exe -m cutout_lab evaluate
.\.venv\Scripts\python.exe -m cutout_lab report
```

Seeds: 42, 43, 44. Runs are interleaved by seed, not selected by performance.
Metrics are percentages; sample standard deviation is computed across three training seeds.
Primary endpoint: clean test accuracy. Secondary: mean accuracy across lengths 8/16/24,
each averaged over three shared deterministic test placements.
Test squares are wholly inside the image and therefore differ from boundary-clipped training squares.
No test-set tuning. No statistical significance is claimed from three seeds alone.

## Evidence and outputs

Each run contains configuration, provenance, initialization hash, per-epoch CSV,
completion status, validation-selected best weights, and resumable last checkpoint.
`evaluation.json` and `predictions/*.csv` contain all clean and occluded predictions.
`output/summary.json` is the sole numeric input for the paper builder.
`output/audit.json` checks complete runs, identical initialization, weight hashes, and prediction-derived accuracy.
Figures are emitted as PNG and vector PDF. Tables are CSV.

The clean train probe comprises 100 training samples per category selected before training.
Its train-validation gap is descriptive and is not the full training-set generalization gap.
Failure cases use seed 42 and ascending test indices, first three improvements and regressions.

## Weights and release

Formal training is in progress. The generated release manifest will list every method/seed,
best epoch, exact metrics and SHA-256. The final repository includes a `weights/` mirror;
the same files are also packaged for GitHub Releases. `scripts/package_release.py` refuses incomplete experiments.
Do not interpret pilot or synthetic unit-test artifacts as formal experiment results.

## Data download

Original source: https://www.cs.toronto.edu/~kriz/cifar.html

Archive MD5: `c58f30108f718f92721af3b95e74349a` (170,498,071 bytes).
If the original server is unavailable, a byte-identical mirror may be used only after verifying this MD5.
The local preparation used a verified archive mirror at
https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python ; extracted batch files are also checked by torchvision.
The original dataset and Python environment are not included in the repository.

## Attribution

The model and augmentation code are independently implemented from the papers, not copied from upstream repositories.
The design is a course reproduction and controlled evaluation; progressive probability is not claimed as a historically novel algorithm.

- He et al. Deep Residual Learning for Image Recognition. CVPR 2016. https://arxiv.org/abs/1512.03385
- DeVries and Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. 2017. https://arxiv.org/abs/1708.04552
- Original Cutout repository: https://github.com/uoguelph-mlrg/Cutout
- Zhong et al. Random Erasing Data Augmentation. https://arxiv.org/abs/1708.04896
- Bengio et al. Curriculum Learning. ICML 2009. https://doi.org/10.1145/1553374.1553380
- Krizhevsky. Learning Multiple Layers of Features from Tiny Images. 2009.
- Loshchilov and Hutter. SGDR. ICLR 2017. https://arxiv.org/abs/1608.03983

Code license: MIT. Third-party datasets, papers, fonts and packages retain their respective licenses.
