# -*- coding: utf-8 -*-
"""01 数据准备：把 Food-101 的 parquet 分片解成按类别分目录的 jpg

数据来源：HuggingFace 镜像 ethz/food101（Food-101，ETH Zurich，仅限研究/教学用途）
本脚本只保留 train 分片里出现过的类别，并把 validation 分片里属于这些类别的图片作为测试集。

输出：
    data/images/train/<类别名>/*.jpg
    data/images/test/<类别名>/*.jpg
    data/labels.json     类别名列表 + 每类张数
"""
import io
import json
import os
import sys

import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PQ_DIR = os.path.join(BASE, "data", "food101")
OUT_TRAIN = os.path.join(BASE, "data", "images", "train")
OUT_TEST = os.path.join(BASE, "data", "images", "test")
os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_TEST, exist_ok=True)

# Food-101 的 101 个类名（与 parquet 里 label 的整数索引一一对应）
CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad",
    "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad",
    "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheesecake", "cheese_plate",
    "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse",
    "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame",
    "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
    "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt",
    "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon",
    "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros",
    "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese",
    "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck", "pho", "pizza",
    "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli",
    "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops", "seaweed_salad",
    "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls",
    "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles",
]


def dump(df, out_dir, tag, keep_labels):
    n = 0
    for _, row in df.iterrows():
        lab = int(row["label"])
        if lab not in keep_labels:
            continue
        name = CLASSES[lab]
        cls_dir = os.path.join(out_dir, name)
        os.makedirs(cls_dir, exist_ok=True)
        im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        im.save(os.path.join(cls_dir, "%s_%06d.jpg" % (tag, n)), quality=92)
        n += 1
        if n % 3000 == 0:
            print("  %s 已导出 %d 张" % (tag, n))
    return n


print("读取 train 分片 ...")
train_files = sorted(f for f in os.listdir(PQ_DIR) if f.startswith("train"))
tr = pd.concat([pd.read_parquet(os.path.join(PQ_DIR, f)) for f in train_files], ignore_index=True)
keep = sorted(tr["label"].unique())
print("train 张数:", len(tr), " 类别数:", len(keep))

print("读取 validation 分片 ...")
val_files = sorted(f for f in os.listdir(PQ_DIR) if f.startswith("validation"))
va = pd.concat([pd.read_parquet(os.path.join(PQ_DIR, f)) for f in val_files], ignore_index=True)
print("validation 张数:", len(va))

print("\n导出训练集 ...")
n_train = dump(tr, OUT_TRAIN, "train", set(keep))
print("\n导出测试集（只用与训练集相同的类别）...")
n_test = dump(va, OUT_TEST, "test", set(keep))

labels = {
    "classes": [CLASSES[i] for i in keep],
    "label_ids": [int(i) for i in keep],
    "n_train": n_train,
    "n_test": n_test,
    "source": "HuggingFace ethz/food101 (Food-101, ETH Zurich, research/teaching use only)",
}
with open(os.path.join(BASE, "data", "labels.json"), "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)

print("\n完成：train %d 张, test %d 张, 类别 %d 个" % (n_train, n_test, len(keep)))
print("类别:", ", ".join(labels["classes"]))
