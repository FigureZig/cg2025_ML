from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from app.core.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, default=1, index=True)

    ref_no = Column(String, index=True, nullable=True)
    trx_date = Column(DateTime(timezone=True), index=True, nullable=False)
    amount = Column(Float, nullable=False)
    type = Column(String, nullable=False)
    balance_after = Column(Float, nullable=True)

    category = Column(String, default="Misc", index=True)

    is_anomaly = Column(Boolean, default=False)
    anomaly_reason = Column(Text, nullable=True)

    is_corrected = Column(Boolean, default=False)
    original_category = Column(String, nullable=True)
    is_manual = Column(Boolean, default=False)


class Budget(Base):
    __tablename__ = "budgets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, default=1, index=True)

    category = Column(String, unique=True, index=True, nullable=False)
    amount_limit = Column(Float, nullable=False, default=0.0)