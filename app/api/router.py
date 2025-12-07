from fastapi import APIRouter, Depends, HTTPException, Query, Body, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from datetime import datetime
from pydantic import BaseModel

from app.core.database import get_db
from app.schemas.transaction import TransactionCreate, TransactionResponse, TransactionUpdateCategory
from app.schemas.advisory import AdvisoryResponse, GoalRequest
from app.schemas.budget import BudgetSet, BudgetResponse
from app.schemas.analytics import FinancialHealthResponse
from app.services.finance import FinanceService
from app.ml.engine import ml_engine
from app.config import settings

api_router = APIRouter()


class BalanceResponse(BaseModel):
    balance: float
    last_transaction_date: datetime | None
    system_date: datetime


@api_router.post("/ml/analyze", tags=["ML Engine"])
async def analyze_transaction(
        amount: float = Body(..., example=5000),
        ref_no: str = Body("", example="MCDONALDS"),
        trx_date: datetime = Body(None)
):
    if not trx_date:
        trx_date = settings.MOCK_NOW

    simulated_trx = {
        "amount": amount,
        "ref_no": ref_no,
        "trx_date": trx_date,
        "type": "withdrawal",
        "deposit": 0,
        "withdrawal": amount
    }
    return ml_engine.predict_transaction(simulated_trx)


@api_router.post("/ml/calculate-strategy", response_model=AdvisoryResponse, tags=["ML Engine"])
async def calculate_strategy(req: GoalRequest, db: AsyncSession = Depends(get_db)):
    goals = [{"amount": req.goal_amount, "date": req.goal_date, "name": req.goal_name}]
    res = await FinanceService.generate_strategy(db, goals)
    if not res:
        return {
            "meta": {"status": "ERROR", "goal_gap": 0, "current_balance": 0},
            "advice": {"message": "Недостаточно данных для прогноза", "steps": []},
            "simulation": {"events": [], "graph": []}
        }
    return res


@api_router.post("/ml/retrain", tags=["ML Engine"])
async def retrain_system(
        bg_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_db)
):
    df = await FinanceService.get_full_history_df(db)
    if df.empty or len(df) < 10:
        return {"status": "skipped", "reason": "Not enough data (<10 samples)"}

    bg_tasks.add_task(ml_engine.retrain_model, df)
    return {"status": "accepted", "message": "Retraining started in background"}


@api_router.get("/balance", response_model=BalanceResponse, tags=["Transactions"])
async def get_balance(db: AsyncSession = Depends(get_db)):
    return await FinanceService.get_current_balance(db)


@api_router.post("/transactions", response_model=TransactionResponse, tags=["Transactions"])
async def add_transaction(trx: TransactionCreate, db: AsyncSession = Depends(get_db)):
    return await FinanceService.process_transaction(db, trx.model_dump())


@api_router.get("/transactions", response_model=List[TransactionResponse], tags=["Transactions"])
async def get_history(skip: int = 0, limit: int = 50, db: AsyncSession = Depends(get_db)):
    return await FinanceService.get_history(db, skip, limit)


@api_router.patch("/transactions/{transaction_id}/correct", response_model=TransactionResponse, tags=["Transactions"])
async def correct_category(transaction_id: int, correction: TransactionUpdateCategory,
                           db: AsyncSession = Depends(get_db)):
    result = await FinanceService.correct_category(db, transaction_id, correction)
    if not result:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return result


@api_router.get("/budgets", response_model=List[BudgetResponse], tags=["Budgets"])
async def get_budgets(db: AsyncSession = Depends(get_db)):
    return await FinanceService.get_budgets_status(db)


@api_router.post("/budgets", response_model=BudgetResponse, tags=["Budgets"])
async def set_budget(budget: BudgetSet, db: AsyncSession = Depends(get_db)):
    return await FinanceService.set_category_limit(db, budget.category, budget.amount_limit)


@api_router.get("/analytics/summary", tags=["Analytics"])
async def get_analytics_summary(
        period: str = Query(..., regex="^(week|month|year|all)$"),
        db: AsyncSession = Depends(get_db)
):
    return await FinanceService.get_analytics_summary(db, period)


@api_router.get("/analytics/monthly", tags=["Analytics"])
async def get_monthly_analytics(db: AsyncSession = Depends(get_db)):
    return await FinanceService.get_monthly_stats(db)


@api_router.get("/analytics/health", response_model=FinancialHealthResponse, tags=["Analytics"])
async def get_health_score(db: AsyncSession = Depends(get_db)):
    return await FinanceService.calculate_financial_health_score(db)


@api_router.get("/categories", tags=["System"])
async def get_categories(db: AsyncSession = Depends(get_db)):
    return await FinanceService.get_all_categories(db)


@api_router.delete("/reset", tags=["System"])
async def hard_reset(db: AsyncSession = Depends(get_db)):
    return await FinanceService.hard_reset(db)