from pydantic import BaseModel, ConfigDict

class FinancialHealthResponse(BaseModel):
    score: int
    status: str

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "score": 78,
            "status": "Good"
        }
    })