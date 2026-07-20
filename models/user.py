from __future__ import annotations

import json
from datetime import datetime
from typing import List

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase

MEAL_HISTORY_MAX = 20

# 允许通过 update_user_profile 修改的字段白名单
UPDATABLE_FIELDS = {"likes", "dislikes", "budget", "diet_restriction"}
LIST_FIELDS = {"likes", "dislikes", "meal_history"}
VALID_BUDGETS = {"low", "mid", "high"}


class Base(DeclarativeBase):
    pass


class UserProfileORM(Base):
    __tablename__ = "user_profiles"

    user_id = Column(String(64), primary_key=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 列表字段以 JSON 字符串存储
    likes = Column(Text, default="[]")         # list[str]
    dislikes = Column(Text, default="[]")      # list[str]
    meal_history = Column(Text, default="[]")  # list[str], max 20

    # 标量字段
    budget = Column(String(8), default=None)           # low / mid / high；None 表示用户从未设置
    diet_restriction = Column(String(64), default="")  # 素食 / 清真 / 无
    category_preferences = Column(Text, default=None)  # JSON 字符串，如 {"prefer": ["川菜"], "avoid": []}


# ---------- 序列化工具 ----------

def decode_list(raw: str | None) -> List[str]:
    """将数据库中存储的 JSON 字符串解码为 Python list。"""
    if not raw:
        return []
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def encode_list(items: List[str]) -> str:
    return json.dumps(items, ensure_ascii=False)


def orm_to_dict(row: UserProfileORM) -> dict:
    cat_raw = row.category_preferences
    try:
        category_preferences = json.loads(cat_raw) if cat_raw else {}
    except json.JSONDecodeError:
        category_preferences = {}

    return {
        "user_id": row.user_id,
        "likes": decode_list(row.likes),
        "dislikes": decode_list(row.dislikes),
        "meal_history": decode_list(row.meal_history),
        "budget": row.budget,
        "diet_restriction": row.diet_restriction or "",
        "category_preferences": category_preferences,
    }
