# -*- coding: utf-8 -*-
"""04 结果可视化：混淆矩阵 + 每类指标 + 热量误差分布

产出（results/ 目录）：
    confusion_matrix.png   两个模型的归一化混淆矩阵对比
    per_class_metrics.png  每类 precision/recall/f1 条形图（迁移学习）
    calorie_error.png      热量误差分布直方图 + 误差最大的类别
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
with open(os.path.join(RES, "metrics.json"), encoding="utf-8") as f:
    M = json.load(f)
CLASSES = M["classes"]
N = len(CLASSES)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 混淆矩阵 ----------------
cms = {}
for tag in ("baseline_cnn", "mobilenetv2"):
    p = os.path.join(RES, "cm_%s.npy" % tag)
    if os.path.exists(p):
        cms[tag] = np.load(p)

if cms:
    fig, axes = plt.subplots(1, len(cms), figsize=(9 * len(cms), 8))
    if len(cms) == 1:
        axes = [axes]
    for ax, (tag, cm) in zip(axes, cms.items()):
        cmn = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title("Normalized confusion matrix — %s" % tag)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(CLASSES, rotation=90, fontsize=7)
        ax.set_yticklabels(CLASSES, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "confusion_matrix.png"), dpi=150)
    print("已保存 confusion_matrix.png")

# ---------------- 每类指标（迁移学习）----------------
best = next(r for r in M["results"] if r["tag"] == "mobilenetv2")
f1 = best["per_class_f1"]
rec = best["per_class_recall"]
order = np.argsort(f1)[::-1]

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(N)
ax.bar(x - 0.2, [f1[i] for i in order], width=0.4, label="F1")
ax.bar(x + 0.2, [rec[i] for i in order], width=0.4, label="Recall")
ax.set_xticks(x)
ax.set_xticklabels([CLASSES[i] for i in order], rotation=60, ha="right", fontsize=8)
ax.set_ylabel("score")
ax.set_title("Per-class F1 / Recall — MobileNetV2 (Top-1 %.3f, macro-F1 %.3f)" % (
    best["top1"], best["macro_f1"]))
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RES, "per_class_metrics.png"), dpi=150)
print("已保存 per_class_metrics.png")

# ---------------- 热量误差 ----------------
cp = os.path.join(RES, "calorie_eval.json")
if os.path.exists(cp):
    with open(cp, encoding="utf-8") as f:
        C = json.load(f)
    pc = C["per_class_mae"]
    names = sorted(pc, key=lambda k: -pc[k])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(names, [pc[k] for k in names], color="#d62728", alpha=0.8)
    ax.set_ylabel("MAE (kcal)")
    ax.set_title("Per-class calorie MAE — simulated condition (overall %.1f kcal)" % C["mae_simulated_kcal"])
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "calorie_error.png"), dpi=150)
    print("已保存 calorie_error.png")

print("\n结果文件：")
for f in sorted(os.listdir(RES)):
    print("  ", f, "%.0f KB" % (os.path.getsize(os.path.join(RES, f)) / 1024))
