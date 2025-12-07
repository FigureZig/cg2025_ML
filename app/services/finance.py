import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, delete, text, case, and_
from datetime import datetime, timedelta

from app.models.transaction import Transaction, Budget
from app.ml.engine import ml_engine
from app.schemas.transaction import TransactionUpdateCategory, TransactionResponse
from app.schemas.budget import BudgetResponse
from app.config import settings


class FinanceService:
    @staticmethod
    def get_system_time() -> datetime:
        return settings.MOCK_NOW

    @staticmethod
    async def process_transaction(db: AsyncSession, trx_data: dict) -> TransactionResponse:
        if not trx_data.get('trx_date'):
            trx_data['trx_date'] = FinanceService.get_system_time()

        # 1. ML Predict
        prediction = ml_engine.predict_transaction(trx_data)
        category = prediction["category"]

        # 2. Limit Check Logic
        warning_msg = None
        if trx_data['type'] == 'withdrawal':
            current_month_start = trx_data['trx_date'].replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            # Sum spent in this category for current month
            spent_query = select(func.sum(Transaction.amount)).where(
                and_(
                    Transaction.category == category,
                    Transaction.type == 'withdrawal',
                    Transaction.trx_date >= current_month_start
                )
            )
            spent_res = await db.execute(spent_query)
            already_spent = spent_res.scalar() or 0.0

            # Get Budget
            limit_query = select(Budget).where(Budget.category == category)
            limit_res = await db.execute(limit_query)
            budget = limit_res.scalar_one_or_none()

            if budget and budget.amount_limit > 0:
                total_projected = already_spent + trx_data['amount']
                if total_projected > budget.amount_limit:
                    warning_msg = f"ПРЕВЫШЕНИЕ ЛИМИТА! ({total_projected:.0f} / {budget.amount_limit:.0f})"
                elif total_projected > (budget.amount_limit * 0.9):
                    warning_msg = f"Внимание: исчерпано 90% лимита ({total_projected:.0f} / {budget.amount_limit:.0f})"

        # 3. Create DB Object
        db_obj = Transaction(
            amount=trx_data['amount'],
            type=trx_data['type'],
            trx_date=trx_data['trx_date'],
            ref_no=trx_data.get('ref_no'),
            is_manual=trx_data.get('is_manual', False),
            category=category,
            is_anomaly=prediction["is_anomaly"],
            anomaly_reason=prediction["anomaly_reason"]
        )

        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)

        # 4. Construct Response manually to inject warning
        return TransactionResponse(
            id=db_obj.id,
            amount=db_obj.amount,
            type=db_obj.type,
            trx_date=db_obj.trx_date,
            ref_no=db_obj.ref_no,
            is_manual=db_obj.is_manual,
            category=db_obj.category,
            is_anomaly=db_obj.is_anomaly,
            anomaly_reason=db_obj.anomaly_reason,
            warning_message=warning_msg,
            is_corrected=db_obj.is_corrected,
            original_category=db_obj.original_category
        )

    @staticmethod
    async def get_budgets_status(db: AsyncSession) -> list[BudgetResponse]:
        now = FinanceService.get_system_time()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Get all distinct categories
        cats_query = select(Transaction.category).distinct()
        cats_res = await db.execute(cats_query)
        active_cats = set(r for r in cats_res.scalars().all())
        default_cats = {"Food", "Transport", "Shopping", "Rent", "Salary", "Misc"}
        all_cats = active_cats.union(default_cats)

        result = []
        for cat in all_cats:
            # Get limit
            b_query = select(Budget).where(Budget.category == cat)
            b_res = await db.execute(b_query)
            budget = b_res.scalar_one_or_none()
            limit = budget.amount_limit if budget else 0.0

            # Get spent
            s_query = select(func.sum(Transaction.amount)).where(
                and_(
                    Transaction.category == cat,
                    Transaction.type == 'withdrawal',
                    Transaction.trx_date >= month_start
                )
            )
            s_res = await db.execute(s_query)
            spent = s_res.scalar() or 0.0

            if limit > 0:
                pct = (spent / limit) * 100
                rem = limit - spent
            else:
                pct = 0.0
                rem = 0.0

            result.append(BudgetResponse(
                category=cat,
                amount_limit=limit,
                spent_in_current_month=spent,
                percentage_used=round(pct, 1),
                remaining=round(rem, 2)
            ))

        return sorted(result, key=lambda x: x.percentage_used, reverse=True)

    @staticmethod
    async def set_category_limit(db: AsyncSession, category: str, amount: float):
        stmt = select(Budget).where(Budget.category == category)
        res = await db.execute(stmt)
        budget = res.scalar_one_or_none()

        if budget:
            budget.amount_limit = amount
        else:
            budget = Budget(category=category, amount_limit=amount)
            db.add(budget)

        await db.commit()
        await db.refresh(budget)

        now = FinanceService.get_system_time()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        s_query = select(func.sum(Transaction.amount)).where(
            and_(
                Transaction.category == category,
                Transaction.type == 'withdrawal',
                Transaction.trx_date >= month_start
            )
        )
        s_res = await db.execute(s_query)
        spent = s_res.scalar() or 0.0

        if budget.amount_limit > 0:
            pct = (spent / budget.amount_limit) * 100
            rem = budget.amount_limit - spent
        else:
            pct = 0.0
            rem = 0.0

        return BudgetResponse(
            category=budget.category,
            amount_limit=budget.amount_limit,
            spent_in_current_month=spent,
            percentage_used=round(pct, 1),
            remaining=round(rem, 2)
        )

    @staticmethod
    async def get_history(db: AsyncSession, skip: int = 0, limit: int = 50):
        current_time = FinanceService.get_system_time()
        query = (
            select(Transaction)
            .where(Transaction.trx_date <= current_time)
            .order_by(desc(Transaction.trx_date))
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(query)
        return result.scalars().all()

    @staticmethod
    async def get_current_balance(db: AsyncSession):
        current_time = FinanceService.get_system_time()
        # Find last transaction to get cumulative balance
        query = (
            select(Transaction)
            .where(Transaction.trx_date <= current_time)
            .order_by(desc(Transaction.trx_date), desc(Transaction.id))
            .limit(1)
        )
        result = await db.execute(query)
        last_trx = result.scalar_one_or_none()

        real_balance = last_trx.balance_after if (last_trx and last_trx.balance_after is not None) else 0.0
        last_date = last_trx.trx_date if last_trx else None

        return {
            "balance": round(real_balance, 2),
            "last_transaction_date": last_date,
            "system_date": current_time
        }

    @staticmethod
    async def get_analytics_summary(db: AsyncSession, period: str):
        now = FinanceService.get_system_time()
        if period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        elif period == 'year':
            start_date = now - timedelta(days=365)
        else:
            start_date = datetime(1970, 1, 1)

        query = select(
            func.sum(case((Transaction.type == 'withdrawal', Transaction.amount), else_=0)).label('expenses'),
            func.sum(case((Transaction.type == 'deposit', Transaction.amount), else_=0)).label('income')
        ).where(
            and_(Transaction.trx_date >= start_date, Transaction.trx_date <= now)
        )
        res = await db.execute(query)
        row = res.one()

        expenses = row.expenses or 0.0
        income = row.income or 0.0

        cat_query = select(
            Transaction.category,
            func.sum(Transaction.amount).label('total')
        ).where(
            and_(
                Transaction.trx_date >= start_date,
                Transaction.trx_date <= now,
                Transaction.type == 'withdrawal'
            )
        ).group_by(Transaction.category).order_by(desc('total'))

        cat_res = await db.execute(cat_query)
        breakdown = {r.category: round(r.total, 2) for r in cat_res.all()}

        return {
            "period": period,
            "start_date": start_date,
            "end_date": now,
            "expenses": round(expenses, 2),
            "income": round(income, 2),
            "net_flow": round(income - expenses, 2),
            "breakdown": breakdown
        }

    @staticmethod
    async def get_monthly_stats(db: AsyncSession, months: int = 6):
        end_date = FinanceService.get_system_time()
        start_date = end_date - timedelta(days=30 * months)

        query = select(
            func.strftime('%Y-%m', Transaction.trx_date).label('month'),
            Transaction.type,
            func.sum(Transaction.amount).label('total')
        ).where(
            and_(Transaction.trx_date >= start_date, Transaction.trx_date <= end_date)
        ).group_by(
            func.strftime('%Y-%m', Transaction.trx_date),
            Transaction.type
        ).order_by(text('month ASC'))

        result = await db.execute(query)
        rows = result.all()

        stats = {}
        for r in rows:
            m, t, amt = r.month, r.type, r.total
            if m not in stats: stats[m] = {"income": 0, "expense": 0}
            if t == 'deposit':
                stats[m]['income'] = amt
            else:
                stats[m]['expense'] = amt

        months_labels = sorted(stats.keys())
        incomes = [round(stats[m]['income'], 2) for m in months_labels]
        expenses = [round(stats[m]['expense'], 2) for m in months_labels]
        avg_expense = sum(expenses) / len(expenses) if expenses else 0

        return {
            "labels": months_labels,
            "incomes": incomes,
            "expenses": expenses,
            "avg_monthly_expense": round(avg_expense, 2)
        }

    @staticmethod
    async def correct_category(db: AsyncSession, transaction_id: int, correction: TransactionUpdateCategory):
        query = select(Transaction).where(Transaction.id == transaction_id)
        result = await db.execute(query)
        transaction = result.scalar_one_or_none()
        if not transaction: return None

        transaction.original_category = transaction.category
        transaction.category = correction.category
        transaction.is_corrected = True
        transaction.is_manual = True

        await db.commit()
        await db.refresh(transaction)
        return transaction

    @staticmethod
    async def get_all_categories(db: AsyncSession):
        q = select(Transaction.category).distinct()
        res = await db.execute(q)
        db_cats = [r for r in res.scalars().all()]
        default_cats = ["Food", "Transport", "Shopping", "Rent", "Salary", "Misc", "Health"]
        return list(set(db_cats + default_cats))

    @staticmethod
    async def hard_reset(db: AsyncSession):
        await db.execute(delete(Transaction))
        await db.execute(delete(Budget))
        await db.commit()
        return {"status": "cleared", "timestamp": FinanceService.get_system_time()}

    @staticmethod
    async def generate_strategy(db: AsyncSession, goals: list):
        df = await FinanceService.get_full_history_df(db)
        if df.empty: return None
        return ml_engine.generate_strategy(df, goals=goals, current_date=FinanceService.get_system_time())

    @staticmethod
    async def get_full_history_df(db: AsyncSession) -> pd.DataFrame:
        current_time = FinanceService.get_system_time()
        query = select(Transaction).where(Transaction.trx_date <= current_time).order_by(Transaction.trx_date)
        result = await db.execute(query)
        transactions = result.scalars().all()
        if not transactions: return pd.DataFrame()

        df = pd.DataFrame([t.__dict__ for t in transactions])
        if "_sa_instance_state" in df.columns:
            del df["_sa_instance_state"]
        return df