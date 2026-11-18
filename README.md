> **在线仓库**：https://github.com/xy-yyds6/food-calorie-agent

> 本仓库不包含原始图像数据（Food-101 分片 1.3 GB + 解压出的 jpg 约 500 MB），
> 按下面第三节的两条命令可以完全重建；仓库里保留了训练好的模型与全部实验结果。
# 饮食热量记录与估算 Agent（FoodCalorieAgent）

拍一张餐盘照片，Agent 识别食物、追问分量和做法，估算这一餐的热量并累计到当日摄入。

- **机器学习组件**：迁移学习图像分类（MobileNetV2，ImageNet 预训练 + 微调），26 类食物
- **Agent**：有限状态机 + 固定选项交互（不开放自由对话），低置信度时转人工确认
- **课程内容**：卷积网络、迁移学习、多分类评估（Top-1/Top-5/macro-F1）、回归（热量估算）

## 一、目录结构

```
foodagent/
├─ scripts/
│  ├─ 01_prepare_data.py        # parquet -> 按类别分目录的 jpg
│  ├─ 02_train.py               # 基线小 CNN + MobileNetV2 迁移学习
│  ├─ 03_evaluate_calorie.py    # 分类指标 + 两套热量 MAE
│  ├─ 04_visualize.py           # 混淆矩阵 / 每类指标 / 误差分布
│  ├─ agent.py                  # 状态机 + 营养库 + 当日累计
│  └─ app_cli.py                # 命令行演示（一次完整交互）
├─ data/
│  ├─ food101/                  # 原始 parquet 分片（3 个，约 1.3 GB）
│  ├─ images/train|test/        # 解出来的 jpg（26 类）
│  ├─ labels.json               # 类别列表与张数
│  └─ nutrition.json            # 营养库（热量密度 + 份量档位 + 烹饪系数）
├─ models/                      # baseline_cnn.keras / mobilenetv2_transfer.keras
└─ results/                     # 指标 json + 训练曲线 + 混淆矩阵 + 误差图
```

## 二、环境

```bash
conda create -n food python=3.10 -y
conda activate food
pip install -r requirements.txt
```

本机实测环境：Python 3.10.11、TensorFlow 2.15.1、Keras 2.15.0、Pillow、pandas、pyarrow、matplotlib。

## 三、数据来源与许可

| 数据 | 来源 | 规模 | 许可 |
|---|---|---|---|
| Food-101 | HuggingFace `ethz/food101`（Food-101，ETH Zurich） | 3 个 parquet 分片，26 类共 18938 训练 + 6500 测试 | 仅限研究与教学用途（dataset card 标注 `license: unknown`，按原数据集的使用条款） |
| 营养数据 | 整理自 USDA FoodData Central 公开数据 | 26 类食物的 kcal/100g + 份量基准 | 公有领域（US Government Work） |

## 四、运行

```bash
# 1 解数据（需要先下载 parquet 到 data/food101/）
python scripts/01_prepare_data.py

# 2 训练（约 40 分钟，CPU）
python scripts/02_train.py

# 3 评估分类与热量误差
python scripts/03_evaluate_calorie.py

# 4 出图
python scripts/04_visualize.py

# 5 演示一次完整交互
python scripts/app_cli.py data/images/test/pizza/test_002500.jpg
```

`data/food101/*.parquet` 的下载地址（HuggingFace 镜像）：

```
https://hf-mirror.com/datasets/ethz/food101/resolve/main/data/train-00000-of-00008.parquet
https://hf-mirror.com/datasets/ethz/food101/resolve/main/data/train-00001-of-00008.parquet
https://hf-mirror.com/datasets/ethz/food101/resolve/main/data/validation-00000-of-00003.parquet
```

## 五、结果

| 模型 | Top-1 | Top-5 | macro-F1 |
|---|---|---|---|
| 基线小 CNN（从零训练） | 6.58% | 26.46% | 2.96% |
| **MobileNetV2 迁移学习** | **55.26%** | **91.51%** | **54.30%** |

| 热量口径 | MAE |
|---|---|
| 模拟条件（只含分类误差） | 69.5 kcal |
| 真实交互（用户把中份估成小份） | 134.9 kcal |

低置信度（<0.55）样本占 **15.2%**，这部分会触发人工确认。

## 六、复用组件清单

| 组件 | 版本 | 来源 | 许可 | 用途 |
|---|---|---|---|---|
| TensorFlow / Keras | 2.15.1 / 2.15.0 | https://www.tensorflow.org/ | Apache-2.0 | 模型训练与推理 |
| MobileNetV2 预训练权重 | ImageNet | `tf.keras.applications` | Apache-2.0 | 迁移学习主干 |
| Food-101 | — | https://huggingface.co/datasets/ethz/food101 | 研究/教学用途 | 图像分类数据 |
| USDA FoodData Central | — | https://fdc.nal.usda.gov/ | 公有领域 | 食物热量密度 |
| Pillow | 10.x | https://python-pillow.org/ | HPND | 图像读取（支持中文路径） |
| pandas / pyarrow | 2.3.3 / 25.0.1 | PyPI | BSD-3 / Apache-2.0 | 读取 parquet |

## 七、已知边界

- 训练集为 Food-101（以西餐/快餐为主），**中餐家常菜的识别能力未验证**；
- 分类器为单标签模型，**一盘多种食物混在一起时只能给出一个主类别**；
- 热量由“营养库 × 份量档位 × 烹饪系数”计算，**图像本身不估计克重**，重量依赖用户选择；
- 系统仅作饮食记录参考，不作为医疗或临床营养依据。

