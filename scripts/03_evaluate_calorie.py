# -*- coding: utf-8 -*-
"""03 热量评估：两套 MAE 口径

① 模拟条件 MAE（评估分类误差带来的热量误差）
   测试集每张图：模型预测类别 vs 真实类别，两边都用统一的「中份 + 默认做法」换算热量，
   差值只来自分类错误，不含人的估计噪声。

② 真实交互口径（说明用）
   同样一张图，用户按小/中/大随便选，热量误差会叠加人的主观误差；
   这里用「用户选错一档（中份选成小份或大份）」来模拟，给出误差放大的量化结果。

同时统计：分类正确率、热量误差分布、误差最大的样本、低置信度样本占比。
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
from agent import NutritionDB  # noqa: E402

IMG_SIZE = 160
MODEL = os.path.join(BASE, "models", "mobilenetv2_transfer.keras")
NUTRI = os.path.join(BASE, "data", "nutrition.json")
LABELS = os.path.join(BASE, "data", "labels.json")
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

with open(LABELS, encoding="utf-8") as f:
    CLASSES = json.load(f)["classes"]
db = NutritionDB(NUTRI)

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

model = keras.models.load_model(MODEL)


def load_test():
    X, y, paths = [], [], []
    for ci, cls in enumerate(CLASSES):
        d = os.path.join(BASE, "data", "images", "test", cls)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            im = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            X.append(np.asarray(im, dtype=np.float32))
            y.append(ci)
            paths.append(p)
    return np.stack(X), np.array(y), paths


print("加载测试集 ...")
X, y, paths = load_test()
print("测试样本:", len(X))

prob = model.predict(X, batch_size=64, verbose=0)
pred = prob.argmax(1)
conf = prob.max(1)

# 每类的标准热量（中份 + 默认做法 grilled=1.0）
kcal_std = np.array([db.kcal(c, "medium", "grilled")[1] for c in CLASSES])

true_kcal = kcal_std[y]
pred_kcal = kcal_std[pred]
err = pred_kcal - true_kcal
mae_sim = float(np.mean(np.abs(err)))
acc = float((pred == y).mean())

# 真实交互：假设用户把中份估成小份（0.7 倍）
portion_noise = 0.7
err_real = pred_kcal * portion_noise - true_kcal
mae_real = float(np.mean(np.abs(err_real)))

# 低置信度比例
low_conf = conf < 0.55

print("\n=== 模拟条件（只反映分类误差）===")
print("分类准确率: %.4f" % acc)
print("热量 MAE : %.1f kcal" % mae_sim)
print("误差中位数: %.1f kcal" % float(np.median(np.abs(err))))
print("误差 P90 : %.1f kcal" % float(np.percentile(np.abs(err), 90)))

print("\n=== 真实交互口径（用户把中份估成小份）===")
print("热量 MAE : %.1f kcal（比模拟条件高 %.1f 倍）" % (mae_real, mae_real / max(mae_sim, 1e-6)))

print("\n=== 低置信度 ===")
print("置信度 < 0.55 的样本: %d / %d (%.1f%%)，这些会触发人工确认" % (
    int(low_conf.sum()), len(low_conf), 100.0 * low_conf.mean()))

# 误差最大的样本
order = np.argsort(-np.abs(err))[:10]
print("\n=== 热量误差最大的 10 个样本 ===")
rows = []
for i in order:
    rows.append({"path": os.path.relpath(paths[i], BASE),
                 "true_class": CLASSES[y[i]], "pred_class": CLASSES[pred[i]],
                 "true_kcal": round(float(true_kcal[i]), 1),
                 "pred_kcal": round(float(pred_kcal[i]), 1),
                 "err_kcal": round(float(err[i]), 1),
                 "confidence": round(float(conf[i]), 4)})
    print("  %-16s -> %-16s 真值 %6.1f / 预测 %6.1f 误差 %+7.1f (conf %.2f)" % (
        CLASSES[y[i]], CLASSES[pred[i]], true_kcal[i], pred_kcal[i], err[i], conf[i]))

# 每类误差
print("\n=== 每类平均绝对误差（kcal）===")
per_class = {}
for ci, cls in enumerate(CLASSES):
    m = y == ci
    if m.sum():
        per_class[cls] = round(float(np.abs(err[m]).mean()), 1)
for k, v in sorted(per_class.items(), key=lambda kv: -kv[1])[:8]:
    print("  %-18s %6.1f" % (k, v))

out = {
    "n_test": int(len(X)),
    "accuracy": round(acc, 4),
    "mae_simulated_kcal": round(mae_sim, 1),
    "mae_real_interaction_kcal": round(mae_real, 1),
    "low_confidence_ratio": round(float(low_conf.mean()), 4),
    "per_class_mae": per_class,
    "worst_samples": rows,
}
with open(os.path.join(RES, "calorie_eval.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n已写入 results/calorie_eval.json")
