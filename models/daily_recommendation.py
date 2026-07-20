from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Integer, JSON, String, Text, UniqueConstraint

from models.user import Base


class DailyRecommendationORM(Base):
    __tablename__ = "daily_recommendations"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    user_id           = Column(String(64), nullable=False)
    date              = Column(Date, nullable=False)
    meal_type         = Column(String(16), nullable=False)   # lunch / dinner
    dishes            = Column(JSON, nullable=False)
    ingredient_prices = Column(JSON)
    reasoning         = Column(Text)
    generated_by      = Column(String(16))                   # scheduled/retry/rule/yesterday/lazy
    source_status     = Column(String(16))                   # success/partial/fallback
    is_read           = Column(Integer, default=0)
    created_at        = Column(DateTime, default=datetime.utcnow)
    updated_at        = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "date", "meal_type", name="uk_user_date_meal"),
    )
