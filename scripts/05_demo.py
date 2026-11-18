# -*- coding: utf-8 -*-
"""05 演示：批量跑几张测试图，把完整交互过程导出成演示记录

输出 results/demo_record.txt，可直接截取贴进报告或录屏时对照操作。
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
from app_cli import build_classifier  # noqa: E402
from agent import FoodAgent, NutritionDB  # noqa: E402

RES = os.path.join(BASE, "results")
db = NutritionDB(os.path.join(BASE, "data", "nutrition.json"))
classify = build_classifier()

# 四个演示用例：三个高置信 + 一个低置信（挑一张模型容易搞混的）
CASES = [
    ("data/images/test/pizza", "pizza（高置信，直接记录）", "medium", "baked"),
    ("data/images/test/hamburger", "hamburger（中份，烤）", "medium", "grilled"),
    ("data/images/test/fried_rice", "fried_rice（大份，炒）", "large", "stir_fried"),
    ("data/images/test/tiramisu", "tiramisu（小份，甜品）", "small", "baked"),
]

lines = []
agent = FoodAgent(classify, db)
lines.append("=" * 78)
lines.append("饮食热量记录与估算 Agent —— 演示记录")
lines.append("模型：MobileNetV2 迁移学习（26 类，Top-1 55.26%，Top-5 91.51%）")
lines.append("交互方式：固定选项状态机（分量/做法都是选项，不含自由对话）")
lines.append("=" * 78)

for rel, desc, portion, cooking in CASES:
    d = os.path.join(BASE, rel)
    if not os.path.isdir(d):
        continue
    files = sorted(os.listdir(d))
    if not files:
        continue
    img = os.path.join(d, files[0])
    lines.append("")
    lines.append("-" * 78)
    lines.append("用例：%s" % desc)
    lines.append("输入：%s" % os.path.relpath(img, BASE))
    lines.append("-" * 78)

    resp = agent.submit_image(img)
    topk = agent.pending["topk"]
    lines.append("  [分类] " + ", ".join("%s %.3f" % (k, p) for k, p in topk))
    lines.append("  [Agent] " + resp["ask"])

    if resp["type"] == "need_confirm":
        pick = list(resp["options"])[0]
        lines.append("  [用户] 选择 %s" % resp["options"][pick])
        resp = agent.choose_food(pick)
        lines.append("  [Agent] " + resp["ask"])

    lines.append("  [用户] 选择分量：%s" % portion)
    resp = agent.choose_portion(portion)

    lines.append("  [用户] 选择做法：%s" % cooking)
    resp = agent.choose_cooking(cooking)

    item = resp["item"]
    lines.append("  [结果] %s | %s | %s | %.0f g | %.0f kcal" % (
        item["food_cn"], item["portion"], item["cooking"], item["grams"], item["kcal"]))
    lines.append("  [累计] %.0f / %.0f kcal（剩余 %.0f）" % (
        resp["total_kcal"], resp["target"], resp["remaining"]))
    if resp["warning"]:
        lines.append("  [提醒] " + resp["warning"])

lines.append("")
lines.append("=" * 78)
s = agent.summary()
lines.append("当日汇总：%d 餐，共 %.0f kcal / 目标 %.0f kcal" % (s["meals"], s["total_kcal"], s["target"]))
lines.append("状态机轨迹条数：%d" % len(agent.log))
lines.append("=" * 78)

out = os.path.join(RES, "demo_record.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
print("\n已写入", out)
