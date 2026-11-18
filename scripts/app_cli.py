# -*- coding: utf-8 -*-
"""命令行演示：一次完整的「拍照 -> 追问 -> 记录 -> 累计」流程

用法：
    python app_cli.py 某张食物照片.jpg
    python app_cli.py 照片.jpg --auto medium —-auto 模式自动选默认选项（用于批量演示）

模型：models/mobilenetv2_transfer.keras（训练脚本 02_train.py 产出）
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from agent import FoodAgent, NutritionDB, PORTION_LABEL, COOKING_LABEL  # noqa: E402

IMG_SIZE = 160
MODEL_PATH = os.path.join(os.path.dirname(BASE), "models", "mobilenetv2_transfer.keras")
NUTRITION_PATH = os.path.join(os.path.dirname(BASE), "data", "nutrition.json")


def build_classifier(model_path=MODEL_PATH):
    import tensorflow as tf
    from tensorflow import keras

    model = keras.models.load_model(model_path)
    with open(os.path.join(os.path.dirname(BASE), "data", "labels.json"), encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    def classify(image_path):
        im = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        x = np.asarray(im, dtype=np.float32)[np.newaxis, ...]
        prob = model.predict(x, verbose=0)[0]
        order = np.argsort(-prob)[:3]
        return [(classes[i], float(prob[i])) for i in order]

    return classify


def run_interactive(image_path, auto=None):
    classify = build_classifier()
    db = NutritionDB(NUTRITION_PATH)
    agent = FoodAgent(classify, db)

    print("\n[1] 收到图片：%s" % os.path.basename(image_path))
    resp = agent.submit_image(image_path)
    print("    分类结果：", ", ".join("%s(%.3f)" % (k, p) for k, p in agent.pending["topk"]))
    print("    Agent：", resp["ask"])
    for k, v in resp["options"].items():
        print("        %-10s %s" % (k, v))

    if resp["type"] == "need_confirm":
        choice = auto if auto in resp["options"] else list(resp["options"])[0]
        print("    你选择：%s" % choice)
        resp = agent.choose_food(choice)
        print("    Agent：", resp["ask"])

    # 分量
    portion = auto if auto in PORTION_LABEL else "medium"
    print("    你选择：%s（%s）" % (portion, PORTION_LABEL[portion]))
    resp = agent.choose_portion(portion)
    print("    Agent：", resp["ask"])
    for k, v in resp["options"].items():
        print("        %-10s %s" % (k, v))

    # 做法
    cooking = auto if auto in COOKING_LABEL else "grilled"
    print("    你选择：%s（%s）" % (cooking, COOKING_LABEL[cooking]))
    resp = agent.choose_cooking(cooking)

    print("\n[2] 结果：")
    item = resp["item"]
    print("    %s | %s | %s | %.0f g | %.0f kcal" % (
        item["food_cn"], PORTION_LABEL[item["portion"]], COOKING_LABEL[item["cooking"]],
        item["grams"], item["kcal"]))
    print("    今日累计 %.0f / %.0f kcal（剩余 %.0f）" % (
        resp["total_kcal"], resp["target"], resp["remaining"]))
    if resp["warning"]:
        print("    提醒：", resp["warning"])

    print("\n[3] 交互轨迹（状态机）：")
    for i, t in enumerate(agent.log, 1):
        print("    %2d. %-14s %s" % (i, t["state"], t["event"]))
    return agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="食物照片路径")
    ap.add_argument("--auto", default=None, help="自动选择某个选项（small/medium/large 或 steamed/...）")
    args = ap.parse_args()
    run_interactive(args.image, args.auto)
