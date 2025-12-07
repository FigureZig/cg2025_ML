from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime

class TransactionBase(BaseModel):
    amount: float = Field(..., gt=0)
    type: str = Field(..., pattern="^(withdrawal|deposit)$")
    trx_date: datetime
    ref_no: Optional[str] = None
    is_manual: bool = False

class TransactionCreate(TransactionBase):
    pass

class TransactionResponse(TransactionBase):
    id: int
    category: str
    is_anomaly: bool
    anomaly_reason: Optional[str] = None
    warning_message: Optional[str] = None
    is_corrected: bool
    original_category: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class TransactionUpdateCategory(BaseModel):
    category: str