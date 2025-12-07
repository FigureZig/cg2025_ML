from pydantic import BaseModel, ConfigDict
from typing import List
from datetime import datetime

class AnomalyItem(BaseModel):
    id: int
    date: datetime
    amount: float
    category: str
    title: str
    description: str
    is_read: bool

    model_config = ConfigDict(from_attributes=True)

class AnomalyListResponse(BaseModel):
    total_count: int
    unread_count: int
    items: List[AnomalyItem]

class AnomalyReadRequest(BaseModel):
    is_read: bool