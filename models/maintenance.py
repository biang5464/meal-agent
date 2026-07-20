from __future__ import annotations

import json
from datetime import datetime
from typing import List

from sqlalchemy import Boolean, Column, Date, DateTime, Float, Integer, String, Text

from models.user import Base


class HealthCycleORM(Base):
    __tablename__ = "health_cycles"

    user_id = Column(String(64), primary_key=True, nullable=False)
    cycle_type = Column(String(32), default="menstrual")   # menstrual / custom
    cycle_days = Column(Integer, default=28)
    period_days = Column(Integer, default=5)
    last_start = Column(Date, nullable=True)               # 上次经期开始日期
    symptoms = Column(Text, default="[]")                  # JSON list[str]


class ReminderORM(Base):
    __tablename__ = "reminders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    message = Column(Text, nullable=False)
    reminder_type = Column(String(32), default="diet_check")  # period / restriction_expired / diet_check
    created_at = Column(DateTime, default=datetime.utcnow)
    is_read = Column(Boolean, default=False)


class PreferenceHistoryORM(Base):
    """偏好历史，支持临时限制（expires_at）和时间衰减权重（weight）。"""

    __tablename__ = "preference_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    preference_type = Column(String(32), nullable=False)   # like / dislike / restriction
    value = Column(String(256), nullable=False)
    reason = Column(String(256), default="")
    expires_at = Column(DateTime, nullable=True)           # None 表示永久
    weight = Column(Float, default=1.0)                    # 0.1 – 1.0
    last_reinforced = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


# ---------- 序列化工具 ----------

def decode_symptoms(raw: str | None) -> List[str]:
    if not raw:
        return []
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []
