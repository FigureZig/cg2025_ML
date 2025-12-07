from pydantic import BaseModel, Field, ConfigDict

class BudgetSet(BaseModel):
    category: str
    amount_limit: float = Field(..., ge=0)

class BudgetResponse(BaseModel):
    category: str
    amount_limit: float
    spent_in_current_month: float
    percentage_used: float
    remaining: float

    model_config = ConfigDict(from_attributes=True)