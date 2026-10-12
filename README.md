# FER+ 6 分类表情识别：CNN / VGGNet / ResNet 从零实现与系统对比（独立工程）

基于 **PyTorch 从零实现**（不加载任何预训练权重）三种卷积网络架构，在 **FER+ 6 分类数据集**
（中性 / 快乐 / 惊讶 / 悲伤 / 愤怒 / **恐惧**）上训练，系统对比分类精度、收敛速度与参数量，
并分析残差连接的梯度机制、优化器与学习率调度、SE 注意力，以及类别不平衡（fear 1.58%）的处理。

FER2003，该数据集包含以下 7 类基本表情：愤怒 (Angry)、厌恶 (Disgust)、恐惧 (Fear)、快乐 (Happy)、悲伤 (Sad)、惊讶 (Surprise)、中性 (Neutral)
FER+ 对 FER2013 的标签进行了更精细的重新标注，在原有的 7 类基础上，额外增加了 蔑视 (Contempt)。
本次训练是在FER+数据集基础上，将厌恶、蔑视 2种不常见的分类（在心理咨询过程中，这2个分类不常见）数据剔除，然后进行训练。

**主模型 ResNet-18 测试准确率 90.86% ± 0.05（最高 90.90%），17 组实验全部 ≥ 87.03%。**
完整分析见 [`REPORT.md`](REPORT.md)。

## 一、环境

- Python 3.13 / PyTorch 2.14（CUDA 12.6，单卡 RTX 4090D）
- 训练依赖：`torch` `torchvision` `numpy` `pandas` `matplotlib` `Pillow`
- 演示服务依赖：`fastapi` `uvicorn[standard]` `python-multipart`（见 `requirements.txt`）

> 快速上手可先看 [`说明.md`](说明.md)（分类统计 + 结果 + 目录），服务接口见 [`接口说明.md`](接口说明.md)。

## 二、数据集（已内置在本目录）

```
FER2013Train/  label.csv + 21685 张 48x48 PNG
FER2013Valid/  label.csv +  2841 张
FER2013Test/   label.csv +  2713 张
fer2013new_6class.csv          投票明细（Usage, 图像名, 6 情绪列, unknown）
fer2013new_6class_proba.csv    概率形式投票
fer2013new_6class_统计.md      数据集统计说明
```

`label.csv` 每行格式：

```
图像名,"(0, 0, 48, 48)",neutral, happiness, surprise, sadness, anger, fear, unknown
```

- 标签 = **前 6 个情绪列的 argmax**；已逐行校验 27239 行，**unknown 永不胜出**。
- 类别顺序：`0 中性 / 1 快乐 / 2 惊讶 / 3 悲伤 / 4 愤怒 / 5 恐惧`。
- 占比：中性 34.85%、快乐 32.31%、惊讶 12.71%、悲伤 10.86%、愤怒 7.69%、**恐惧 1.58%**（随机基线 34.6%）。
- 归一化统计（训练集实测）：mean **0.5070**、std **0.2539**（RGB 3 通道）。

## 三、目录结构

```
facefinal/
├─ FER2013{Train,Valid,Test}/   数据（图片 + label.csv）
├─ fer2013new_6class*.csv       投票明细 / 概率 / 统计说明
├─ src/
│  ├─ task.py                   任务配置：数据根、类别、归一化统计、输出目录
│  ├─ datasets.py               数据读取与数据增强
│  ├─ metrics.py                混淆矩阵 / Macro-F1 / 平衡准确率
│  ├─ trainer.py                统一训练器（优化器、调度、AMP、类别权重、评估）
│  ├─ plots.py                  训练曲线、混淆矩阵、梯度范数可视化
│  └─ models/                   cnn.py / vgg.py / resnet.py / se.py
├─ app/                         ★ FastAPI 演示服务
│  ├─ main.py                   接口路由（REST + WebSocket）
│  ├─ engine.py                 模型懒加载、预处理、预测
│  └─ static/index.html         演示页面（图片识别 / 视频实时识别）
├─ experiments.py               实验编排（17 组）
├─ analysis/grad_analysis.py    逐层梯度范数分析（梯度消失 / 退化验证）
├─ analysis/compare_curves.py   对比曲线生成
├─ make_report.py               汇总生成 reports/comparison.{csv,md}
├─ scripts/smoke.py             冒烟测试（参数量、单 epoch 计时）
├─ scripts/class_stats.py       数据分类统计（逐类样本数 / 占比）
├─ scripts/check_api.py         接口 + WebSocket 自检（不占端口）
├─ scripts/check_api_live.py    服务启动后的真实链路自检
├─ checkpoints/                 最优权重 *.pt（17 个）
├─ reports/
│  ├─ results/*.json            17 组完整训练历史与指标
│  ├─ curves/*.png              17 张单实验曲线 + 8 张对比图
│  ├─ confusion_*.png           17 张测试集混淆矩阵
│  ├─ grads/                    梯度范数原始数据
│  └─ comparison.{csv,md}       总表（另有 per_class / summary_by_config）
├─ REPORT.md                    完整实验报告
├─ 说明.md                      项目说明（分类统计 + 结果速查）
├─ 接口说明.md                  REST / WebSocket 协议文档
├─ requirements.txt
├─ 启动服务.bat                 一键启动演示服务（Windows）
└─ README.md
```

## 四、运行

```bash
# 1) 冒烟测试：模型参数量 + 单 epoch 计时
python scripts/smoke.py

# 2) 单个实验 / 全部实验（17 组，约 91 分钟）
python experiments.py --only resnet18
python experiments.py --all

# 3) 汇总总表
python make_report.py

# 4) 对比曲线与残差梯度流分析
python analysis/compare_curves.py
python analysis/grad_analysis.py

# 5) 启动演示服务（图片识别 + 视频实时识别，也可双击 启动服务.bat）
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

已有的 17 组结果与图表**开箱即看**；如需重跑，直接执行上述命令即可（会覆盖同名产物）。

## 五、实验设计（17 组，只改变被考察变量）

| 变量 | 取值 |
|---|---|
| 架构 | CNN(0.42M) / VGG-16(14.85M) / ResNet-18(11.17M) / ResNet-32(23.71M) |
| 残差消融 | 同构同参数量的"关闭恒等短路"对照（18 层与 32 层各一组） |
| 数据增强 | 随机裁剪 + 水平翻转 + 旋转 + 色彩抖动（对照组关闭） |
| 优化器 | Adam(lr=1e-3) vs SGD(momentum=0.9, lr=0.1) |
| 学习率调度 | CosineAnnealing vs StepLR(step=20, γ=0.1) |
| 注意力 | SE 模块（轻量化 ResNet 2.76M → 2.78M） |
| 类别不平衡 | 逆频率类别加权 CE（针对 fear 1.58%） |
| 随机种子 | 42 / 123（主模型、SGD、StepLR、轻量化、轻量化+SE 各两组） |

统一设置：60 epoch、batch=128、weight_decay、AMP(BF16)、**验证集**最优权重回测**测试集**。

## 六、结果速览（详见 `REPORT.md`）

| 配置 | 参数量 | 测试准确率 |
|---|---:|---:|
| ResNet-18 关闭增强 | 11.172 M | 91.49% |
| ResNet-32（残差） | 23.713 M | 91.45% |
| ResNet-18 / SGD（2 种子） | 11.172 M | 91.02% ± 0.60 |
| **ResNet-18 主模型（2 种子）** | 11.172 M | **90.86% ± 0.05**（最高 90.90%） |
| ResNet-18 关闭残差（同构） | 11.172 M | 90.93% |
| VGGNet-16 | 14.852 M | 90.90% |
| ResNet-18 / Adam+StepLR（2 种子） | 11.172 M | 90.71% ± 0.05 |
| 轻量化 ResNet + SE（2 种子） | 2.783 M | 90.49% ± 0.16 |
| 轻量化 ResNet（2 种子） | 2.759 M | 90.43% ± 0.13 |
| ResNet-18 类别加权 CE（fear） | 11.172 M | 89.86%（平衡准确率 87.19%） |
| ResNet-32 关闭残差（退化） | 23.713 M | 88.02% |
| 轻量 CNN | 0.423 M | 87.03% |

关键结论：

1. **ResNet 最优**：同配方下 Macro-F1 比 VGG-16 高 1.7pp 且参数少 25%。
2. **残差的价值在深层**：32 层关残差 → −3.43pp、收敛慢 3.4×、训练准确率卡住（退化）；
   梯度分析：初始化末/首梯度 RMS **478.9×（纯） vs 68.2×（残差）**。
3. **类别不平衡没有免费午餐**：类别加权使 fear 召回 68.75% → 75.00%、平衡准确率 +0.78pp，
   但总体准确率 −1.04pp。
4. **错误结构稳定**：悲伤↔中性互判占全部错误的 39.7%。

## 七、演示服务（FastAPI）

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000      # 或双击 启动服务.bat
```

| 入口 | 地址 | 功能 |
|---|---|---|
| 演示页面 | http://127.0.0.1:8000/ | ① 图片识别（拖拽/随机样例，可选 17 个模型）② 摄像头视频实时识别（8 fps，画面叠加情绪标签） |
| Swagger | http://127.0.0.1:8000/docs | 在线调试 REST 接口 |
| `POST /api/recognize` | multipart `file` + `model` | 单图识别，返回 6 类概率与耗时 |
| `WS /ws/video` | 二进制帧 → JSON | 逐帧返回情绪概率，文本消息可热切换模型 |

- 计算设备与模型清单见 `GET /api/health`、`GET /api/models`；协议细节见 [`接口说明.md`](接口说明.md)。
- 自检：`python scripts/check_api.py`（不占端口）/ `python scripts/check_api_live.py`（服务已启动时）。
