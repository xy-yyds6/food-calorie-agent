# -*- coding: utf-8 -*-
"""饮食热量记录 Agent：状态机 + 分类器 + 营养库 + 当日累计

状态转移：
    WAIT_IMAGE --收到图片--> CLASSIFY
    CLASSIFY --置信度 >= 阈值--> ASK_PORTION
    CLASSIFY --置信度 < 阈值--> ASK_FOOD（给 Top-3 候选让用户选）
    ASK_FOOD --用户选定--> ASK_PORTION
    ASK_PORTION --用户选份量--> ASK_COOKING
    ASK_COOKING --用户选做法--> COMPUTE
    COMPUTE --算完--> ACCUMULATE --> OUTPUT
    OUTPUT --继续记录--> WAIT_IMAGE / 结束

所有交互都是固定选项（不含自由对话），LLM 不参与分类、热量计算和状态跳转。
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 160
CONF_THRESHOLD = 0.55      # 低于这个置信度就问用户
DAILY_TARGET = 2000.0      # 每日目标 kcal

PORTION_LABEL = {"small": "小份", "medium": "中份", "large": "大份"}
COOKING_LABEL = {"steamed": "蒸", "boiled": "煮", "stir_fried": "炒",
                 "deep_fried": "炸", "grilled": "烤", "baked": "烘焙", "raw": "生食"}


class NutritionDB:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.foods = raw["foods"]
        self.portion = raw["_portion_grams"]
        self.cooking = raw["_cooking_factor"]

    def has(self, key):
        return key in self.foods

    def cn_name(self, key):
        return self.foods.get(key, {}).get("name_cn", key)

    def kcal(self, key, portion="medium", cooking="grilled"):
        f = self.foods[key]
        grams = f["base_g"] * self.portion.get(portion, 1.0)
        factor = self.cooking.get(cooking, 1.0)
        return grams, grams * f["kcal_per_100g"] / 100.0 * factor


class FoodAgent:
    """状态机的实现；classifier 是一个可调用对象：image_path -> [(class_key, prob), ...]"""

    def __init__(self, classifier, nutrition, threshold=CONF_THRESHOLD, target=DAILY_TARGET):
        self.classifier = classifier
        self.db = nutrition
        self.threshold = threshold
        self.target = target
        self.state = "WAIT_IMAGE"
        self.pending = {"food": None, "portion": None, "cooking": None, "topk": []}
        self.today = []          # [(food_key, portion, cooking, grams, kcal)]
        self.log = []            # 交互轨迹，便于报告里画流程图/演示

    # ---------------- 内部 ----------------
    def _trace(self, event, **kw):
        self.log.append({"state": self.state, "event": event, **kw})

    def _total(self):
        return sum(m["kcal"] for m in self.today)

    # ---------------- 状态转移 ----------------
    def submit_image(self, image_path):
        """WAIT_IMAGE -> CLASSIFY"""
        assert self.state == "WAIT_IMAGE", "当前状态 %s 不能接收图片" % self.state
        self.state = "CLASSIFY"
        self._trace("image_received", path=os.path.basename(image_path))

        topk = self.classifier(image_path)
        self.pending["topk"] = topk
        best_key, best_prob = topk[0]
        self._trace("classified", top=topk[:3])

        if best_prob >= self.threshold:
            self.pending["food"] = best_key
            self.state = "ASK_PORTION"
            return {"type": "classified", "food": best_key, "food_cn": self.db.cn_name(best_key),
                    "confidence": round(best_prob, 4),
                    "ask": "识别为「%s」（置信度 %.2f）。这份的分量大概是多少？" % (self.db.cn_name(best_key), best_prob),
                    "options": PORTION_LABEL}
        # 低置信度 -> 让用户在 Top-3 里选
        self.state = "ASK_FOOD"
        options = {k: self.db.cn_name(k) for k, _ in topk[:3]}
        return {"type": "need_confirm", "confidence": round(best_prob, 4),
                "ask": "这张照片不太确定（最高置信度只有 %.2f），更接近哪一类？" % best_prob,
                "options": options}

    def choose_food(self, food_key):
        """ASK_FOOD -> ASK_PORTION"""
        assert self.state == "ASK_FOOD", "当前状态 %s 不能选食物" % self.state
        if not self.db.has(food_key):
            return {"type": "error", "msg": "营养库里没有这个食物：%s" % food_key}
        self.pending["food"] = food_key
        self._trace("food_chosen", food=food_key)
        self.state = "ASK_PORTION"
        return {"type": "ask_portion",
                "ask": "好的，记录「%s」。分量大概多少？" % self.db.cn_name(food_key),
                "options": PORTION_LABEL}

    def choose_portion(self, portion):
        """ASK_PORTION -> ASK_COOKING"""
        assert self.state == "ASK_PORTION", "当前状态 %s 不能选分量" % self.state
        if portion not in PORTION_LABEL:
            return {"type": "error", "msg": "分量只能是 %s" % list(PORTION_LABEL)}
        self.pending["portion"] = portion
        self._trace("portion_chosen", portion=portion)
        self.state = "ASK_COOKING"
        return {"type": "ask_cooking",
                "ask": "做法是哪一种？",
                "options": COOKING_LABEL}

    def choose_cooking(self, cooking):
        """ASK_COOKING -> COMPUTE -> ACCUMULATE"""
        assert self.state == "ASK_COOKING", "当前状态 %s 不能选做法" % self.state
        if cooking not in COOKING_LABEL:
            return {"type": "error", "msg": "做法只能是 %s" % list(COOKING_LABEL)}
        self.pending["cooking"] = cooking
        self._trace("cooking_chosen", cooking=cooking)
        self.state = "COMPUTE"

        food = self.pending["food"]
        grams, kcal = self.db.kcal(food, self.pending["portion"], cooking)
        item = {"food": food, "food_cn": self.db.cn_name(food),
                "portion": self.pending["portion"], "cooking": cooking,
                "grams": round(grams, 1), "kcal": round(kcal, 1)}
        self.state = "ACCUMULATE"
        self.today.append(item)
        total = self._total()
        self._trace("meal_added", item=item, total=round(total, 1))

        msg = "已记录：%s（%s，%s）约 %.0f g，%.0f kcal。今日累计 %.0f / %.0f kcal。" % (
            item["food_cn"], PORTION_LABEL[item["portion"]], COOKING_LABEL[item["cooking"]],
            grams, kcal, total, self.target)
        warn = None
        if total > self.target:
            warn = "已超出今日目标 %.0f kcal，后面的餐建议清淡一些。" % (total - self.target)
            msg += " " + warn
        self.state = "OUTPUT"
        self._trace("output", total=round(total, 1), warning=warn)
        self.state = "WAIT_IMAGE"      # 回到初始，等待下一张图
        return {"type": "result", "item": item, "total_kcal": round(total, 1),
                "target": self.target, "message": msg, "warning": warn,
                "remaining": round(self.target - total, 1)}

    # ---------------- 辅助 ----------------
    def reset_day(self):
        self.today.clear()
        self.state = "WAIT_IMAGE"
        self.pending = {"food": None, "portion": None, "cooking": None, "topk": []}

    def summary(self):
        return {"meals": len(self.today), "total_kcal": round(self._total(), 1),
                "target": self.target, "items": self.today}
