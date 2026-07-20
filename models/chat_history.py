from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from models.user import Base


class ChatHistoryORM(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    role = Column(String(16), nullable=False)   # "user" 或 "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
