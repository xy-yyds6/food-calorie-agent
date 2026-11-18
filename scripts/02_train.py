# -*- coding: utf-8 -*-
"""02 训练：基线小 CNN vs 迁移学习（MobileNetV2）

- 输入统一 160x160
- 基线：3 个卷积块 + 全连接，从零训练
- 迁移：ImageNet 预训练 MobileNetV2，先冻结主干训练分类头，再解冻后 30 层微调
- 保存模型、训练曲线、测试集指标（Top-1 / Top-5 / macro-F1 / 混淆矩阵）
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE, "data", "images")
MODEL_DIR = os.path.join(BASE, "models")
RES_DIR = os.path.join(BASE, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

IMG_SIZE = 160
BATCH = 64
SEED = 42
EPOCHS_HEAD = 8      # 冻结主干阶段
EPOCHS_FINE = 5      # 微调阶段

with open(os.path.join(BASE, "data", "labels.json"), encoding="utf-8") as f:
    labels = json.load(f)
CLASSES = labels["classes"]
N_CLASS = len(CLASSES)
print("类别数:", N_CLASS)


def load_split(subset):
    """读成内存数组（PIL 支持中文路径，tf.io.gfile 不支持）"""
    from PIL import Image
    X, y = [], []
    for ci, cls in enumerate(CLASSES):
        d = os.path.join(IMG_DIR, subset, cls)
        if not os.path.isdir(d):
            continue
        files = sorted(os.listdir(d))
        for fn in files:
            im = Image.open(os.path.join(d, fn)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            X.append(np.asarray(im, dtype=np.uint8))
            y.append(ci)
        print("  %s %-16s %d 张" % (subset, cls, len(files)))
    return np.stack(X), np.array(y, dtype=np.int32)


def make_ds(subset, shuffle):
    X, y = load_split(subset)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(2000, seed=SEED)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


print("加载训练集 ...")
train_ds = make_ds("train", True)
print("加载测试集 ...")
test_ds = make_ds("test", False)
print("训练批次数:", len(train_ds), " 测试批次数:", len(test_ds))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)


def build_baseline():
    """基线：从零训练的小 CNN"""
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1. / 255)(inputs)
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(N_CLASS, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer():
    """迁移学习：ImageNet 预训练的 MobileNetV2"""
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(N_CLASS, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_transfer"), base


def evaluate(model, tag):
    """Top-1 / Top-5 / macro-F1 / 混淆矩阵"""
    y_true, y_prob = [], []
    for xb, yb in test_ds:
        y_true.append(yb.numpy())
        y_prob.append(model.predict(xb, verbose=0))
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = y_prob.argmax(1)

    top1 = float((y_pred == y_true).mean())
    top5 = float(np.mean([y_true[i] in np.argsort(-y_prob[i])[:5] for i in range(len(y_true))]))

    cm = np.zeros((N_CLASS, N_CLASS), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    f1s = []
    for c in range(N_CLASS):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    macro_f1 = float(np.mean(f1s))

    metrics = {"tag": tag, "top1": top1, "top5": top5, "macro_f1": macro_f1,
               "n_test": int(len(y_true)), "per_class_f1": f1s, "per_class_recall":
               [float(cm[c, c] / cm[c, :].sum()) if cm[c, :].sum() else 0.0 for c in range(N_CLASS)]}
    print("[%s] Top-1 %.4f  Top-5 %.4f  macro-F1 %.4f" % (tag, top1, top5, macro_f1))
    np.save(os.path.join(RES_DIR, "cm_%s.npy" % tag), cm)
    return metrics


all_metrics = []

# ---------------- 基线 ----------------
print("\n===== 基线小 CNN =====")
base_model = build_baseline()
base_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h1 = base_model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS_HEAD, verbose=2)
base_model.save(os.path.join(MODEL_DIR, "baseline_cnn.keras"))
all_metrics.append(evaluate(base_model, "baseline_cnn"))

# ---------------- 迁移学习 ----------------
print("\n===== 迁移学习 MobileNetV2（冻结主干）=====")
trans_model, base = build_transfer()
trans_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h2 = trans_model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS_HEAD, verbose=2)

print("\n===== 迁移学习（解冻后 30 层微调）=====")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
trans_model.compile(optimizer=keras.optimizers.Adam(1e-4),
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h3 = trans_model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS_FINE, verbose=2)
trans_model.save(os.path.join(MODEL_DIR, "mobilenetv2_transfer.keras"))
all_metrics.append(evaluate(trans_model, "mobilenetv2"))

# ---------------- 曲线 ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(h1.history["accuracy"], label="baseline train")
axes[0].plot(h1.history["val_accuracy"], "--", label="baseline val")
axes[0].set_title("Baseline (small CNN)")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy"); axes[0].legend()

acc = h2.history["accuracy"] + h3.history["accuracy"]
vacc = h2.history["val_accuracy"] + h3.history["val_accuracy"]
axes[1].plot(acc, label="transfer train")
axes[1].plot(vacc, "--", label="transfer val")
axes[1].axvline(len(h2.history["accuracy"]) - 0.5, color="gray", ls=":", label="fine-tune start")
axes[1].set_title("MobileNetV2 transfer")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "training_curves.png"), dpi=150)

with open(os.path.join(RES_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"classes": CLASSES, "img_size": IMG_SIZE, "batch": BATCH,
               "epochs_head": EPOCHS_HEAD, "epochs_fine": EPOCHS_FINE,
               "results": all_metrics}, f, ensure_ascii=False, indent=2)
print("\n结果已写入 results/metrics.json 和 training_curves.png")
