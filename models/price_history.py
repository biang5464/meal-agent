from sqlalchemy import Column, Integer, String, DECIMAL, DateTime, Text, Index
from sqlalchemy.sql import func
from models.user import Base


class PriceHistoryORM(Base):
    __tablename__ = "price_history"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    query_term   = Column(String(128), nullable=False)
    platform     = Column(String(32),  nullable=False)
    product_name = Column(String(256), nullable=True)
    price        = Column(DECIMAL(10, 2), nullable=True)
    unit         = Column(String(16),  nullable=True)
    currency     = Column(String(8),   default="AUD")
    url          = Column(Text,        nullable=True)
    recorded_at  = Column(DateTime,    server_default=func.now())

    __table_args__ = (
        Index("idx_query_platform", "query_term", "platform"),
        Index("idx_recorded_at", "recorded_at"),
    )
