from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional
from datetime import datetime
from app.config import settings

class GraphPoint(BaseModel):
    date: str
    balance: float
    balance_optimized: Optional[float] = None

class SimulationEvent(BaseModel):
    date: str
    type: str
    amount: float

class StrategyStep(BaseModel):
    category: str
    current_monthly: float
    cut_monthly: float
    cut_percent: float
    new_limit_monthly: float
    reason: str

class AdvisoryAdvice(BaseModel):
    message: str
    steps: List[StrategyStep] = []

class SimulationData(BaseModel):
    events: List[SimulationEvent]
    graph: List[GraphPoint]

class AdvisoryMeta(BaseModel):
    status: str
    goal_gap: float = 0.0
    current_balance: float = 0.0

class AdvisoryResponse(BaseModel):
    meta: AdvisoryMeta
    advice: AdvisoryAdvice
    simulation: SimulationData

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "meta": {"status": "ACHIEVABLE_WITH_CUTS", "goal_gap": 12000, "current_balance": 45000},
            "advice": {
                "message": "Сократите расходы на категорию 'Shopping'",
                "steps": []
            },
            "simulation": {
                "events": [],
                "graph": []
            }
        }
    })

class GoalRequest(BaseModel):
    goal_name: str
    goal_amount: float
    goal_date: str

    @field_validator('goal_amount')
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

    @field_validator('goal_date')
    @classmethod
    def date_must_be_future(cls, v: str) -> str:
        try:
            if len(v) == 4 and v.isdigit():
                parsed_date = datetime(int(v), 12, 31)
            else:
                parsed_date = datetime.strptime(v, "%Y-%m-%d")

            if parsed_date <= settings.MOCK_NOW:
                raise ValueError(f"Date must be after system time ({settings.MOCK_NOW.date()})")

            return parsed_date.strftime("%Y-%m-%d")
        except ValueError as e:
            if "Date must be after" in str(e):
                raise e
            raise ValueError('Invalid date format (YYYY-MM-DD)')