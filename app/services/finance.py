import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, delete, text, case, and_
from datetime import datetime, timedelta

from app.models.transaction import Transaction, Budget
from app.ml.engine import ml_engine
from app.schemas.notification import AnomalyListResponse, AnomalyItem
from app.schemas.transaction import TransactionUpdateCategory, TransactionResponse
from app.schemas.budget import BudgetResponse
from app.schemas.analytics import FinancialHealthResponse
from app.config import settings


class FinanceService:
    @staticmethod
    def get_system_time() -> datetime:
        return settings.MOCK_NOW

    @staticmethod
    async def process_transaction(db: AsyncSession, trx_data: dict) -> TransactionResponse:
        if not trx_data.get('trx_date'):
            trx_data['trx_date'] = FinanceService.get_system_time()

        prediction = ml_engine.predict_transaction(trx_data)
        category = prediction["category"]

        warning_msg = None
        if trx_data['type'] == 'withdrawal':
            current_month_start = trx_data['trx_date'].replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            spent_query = select(func.sum(Transaction.amount)).where(
                and_(
                    Transaction.category == category,
                    Transaction.type == 'withdrawal',
                    Transaction.trx_date >= current_month_start
                )
            )
            spent_res = await db.execute(spent_query)
            already_spent = spent_res.scalar() or 0.0

            limit_query = select(Budget).where(Budget.category == category)
            limit_res = await db.execute(limit_query)
            budget = limit_res.scalar_one_or_none()

            if budget and budget.amount_limit > 0:
                total_projected = already_spent + trx_data['amount']
                if total_projected > budget.amount_limit:
                    warning_msg = f"ПРЕВЫШЕНИЕ ЛИМИТА! ({total_projected:.0f} / {budget.amount_limit:.0f})"
                elif total_projected > (budget.amount_limit * 0.9):
                    warning_msg = f"Внимание: исчерпано 90% лимита ({total_projected:.0f} / {budget.amount_limit:.0f})"

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

        cats_query = select(Transaction.category).distinct()
        cats_res = await db.execute(cats_query)
        active_cats = set(r for r in cats_res.scalars().all())
        default_cats = {"Food", "Transport", "Shopping", "Rent", "Salary", "Misc"}
        all_cats = active_cats.union(default_cats)

        result = []
        for cat in all_cats:
            b_query = select(Budget).where(Budget.category == cat)
            b_res = await db.execute(b_query)
            budget = b_res.scalar_one_or_none()
            limit = budget.amount_limit if budget else 0.0

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
    async def calculate_financial_health_score(db: AsyncSession) -> FinancialHealthResponse:
        now = FinanceService.get_system_time()
        start_30d = now - timedelta(days=30)
        start_90d = now - timedelta(days=90)

        cf_query = select(
            func.sum(case((Transaction.type == 'withdrawal', Transaction.amount), else_=0)),
            func.sum(case((Transaction.type == 'deposit', Transaction.amount), else_=0))
        ).where(and_(Transaction.trx_date >= start_30d, Transaction.trx_date <= now))

        cf_res = await db.execute(cf_query)
        expenses_30d, income_30d = cf_res.one()
        expenses_30d = expenses_30d or 0.0
        income_30d = income_30d or 0.0

        avg_exp_query = select(func.sum(Transaction.amount)).where(
            and_(Transaction.type == 'withdrawal', Transaction.trx_date >= start_90d)
        )
        avg_res = await db.execute(avg_exp_query)
        total_exp_90d = avg_res.scalar() or 0.0
        monthly_avg_burn = total_exp_90d / 3.0 if total_exp_90d > 0 else expenses_30d

        last_trx_q = select(Transaction.balance_after).order_by(desc(Transaction.trx_date), desc(Transaction.id)).limit(
            1)
        bal_res = await db.execute(last_trx_q)
        current_balance = bal_res.scalar_one_or_none() or 0.0

        score_accum = 0

        # 1. Statistics (Max 70)
        # Cashflow (Max 30)
        if income_30d > 0:
            savings_rate = (income_30d - expenses_30d) / income_30d
            if savings_rate >= 0.2:
                score_accum += 30
            elif savings_rate >= 0.1:
                score_accum += 20
            elif savings_rate >= 0:
                score_accum += 10

        # Runway (Max 20)
        if monthly_avg_burn > 0:
            runway = current_balance / monthly_avg_burn
            if runway >= 3:
                score_accum += 20
            elif runway >= 1:
                score_accum += 10

        # Limits (Max 20)
        budgets_q = select(Budget)
        b_res = await db.execute(budgets_q)
        budgets = b_res.scalars().all()

        limits_ok = True
        month_start = now.replace(day=1, hour=0, minute=0, second=0)

        for b in budgets:
            c_q = select(func.sum(Transaction.amount)).where(
                and_(Transaction.category == b.category, Transaction.type == 'withdrawal',
                     Transaction.trx_date >= month_start)
            )
            c_r = await db.execute(c_q)
            c_s = c_r.scalar() or 0.0
            if b.amount_limit > 0 and c_s > b.amount_limit:
                limits_ok = False
                break

        if limits_ok:
            score_accum += 20
        else:
            score_accum += 5

            # 2. ML Forecast (Max 30)
        df = await FinanceService.get_full_history_df(db)
        if not df.empty and len(df) > 5:
            try:
                # Run simulation for 30 days without specific goals
                strategy = ml_engine.generate_strategy(df, goals=None, current_date=now)

                if strategy and "simulation" in strategy:
                    graph = strategy["simulation"].get("graph", [])
                    if graph and len(graph) > 1:
                        start_bal = graph[0]["balance"]
                        end_bal = graph[-1]["balance"]

                        if end_bal < 0:
                            score_accum -= 20
                        elif end_bal > start_bal * 1.02:
                            score_accum += 30
                        elif end_bal >= start_bal * 0.95:
                            score_accum += 15
                        else:
                            score_accum += 0
            except:
                pass

        final_score = max(0, min(100, score_accum))

        if final_score >= 80:
            status = "Excellent"
        elif final_score >= 50:
            status = "Good"
        elif final_score >= 30:
            status = "Warning"
        else:
            status = "Critical"

        return FinancialHealthResponse(score=final_score, status=status)

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

        from app.core.seed import seed_data
        await seed_data(db)

        return {"status": "soft_reset_completed", "timestamp": FinanceService.get_system_time()}

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

    @staticmethod
    def _generate_compliance_text(trx: Transaction) -> tuple[str, str]:
        raw_reason = (trx.anomaly_reason or "").lower()
        import re

        ratio_match = re.search(r'x(\d+\.?\d*)', raw_reason)
        times = ratio_match.group(1) if ratio_match else "несколько"

        amount_str = f"{int(trx.amount):,} ₽".replace(",", " ")

        if "поступлений" in raw_reason:
            return (
                "Резкие траты после пополнения",
                f"Вы начали активно тратить сразу после зачисления денег. Расходы в {times} раз выше обычного. Убедитесь, что это запланированные покупки."
            )

        if "новая крупная" in raw_reason:
            return (
                f"Необычная покупка в '{trx.category}'",
                f"Мы заметили операцию на {amount_str}. Раньше вы не тратили так много в этой категории. Это точно вы?"
            )

        if "значительная сумма" in raw_reason or "large amount" in raw_reason:
            return (
                "Крупное списание",
                f"Сумма {amount_str} выбивается из вашей обычной истории трат. Обратите внимание."
            )

        if "интенсивность" in raw_reason or "активность" in raw_reason:
            return (
                "Вы тратите быстрее обычного",
                f"Скорость расходов выросла в {times} раз по сравнению с вашей нормой. Возможно, стоит притормозить?"
            )

        if trx.trx_date.hour < 6 and not (trx.trx_date.hour == 0 and trx.trx_date.minute == 0):
            return (
                "Ночная активность",
                f"Операция на {amount_str} проведена ночью. Если это не вы — срочно заблокируйте карту."
            )

        return (
            "Нетипичная операция",
            f"Эта транзакция в категории '{trx.category}' не похожа на ваши обычные траты. Проверьте детали."
        )

    @staticmethod
    async def get_anomaly_feed(db: AsyncSession, limit: int = 50, offset: int = 0) -> AnomalyListResponse:
        count_query = select(func.count(Transaction.id)).where(Transaction.is_anomaly == True)
        unread_query = select(func.count(Transaction.id)).where(
            and_(Transaction.is_anomaly == True, Transaction.is_read == False)
        )

        total_res = await db.execute(count_query)
        unread_res = await db.execute(unread_query)

        total_count = total_res.scalar() or 0
        unread_count = unread_res.scalar() or 0

        data_query = (
            select(Transaction)
            .where(Transaction.is_anomaly == True)
            .order_by(Transaction.is_read.asc(), desc(Transaction.trx_date))
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(data_query)
        rows = result.scalars().all()

        items = []
        for row in rows:
            title, desc_text = FinanceService._generate_compliance_text(row)
            items.append(AnomalyItem(
                id=row.id,
                date=row.trx_date,
                amount=row.amount,
                category=row.category,
                title=title,
                description=desc_text,
                is_read=row.is_read
            ))

        return AnomalyListResponse(
            total_count=total_count,
            unread_count=unread_count,
            items=items
        )

    @staticmethod
    async def mark_anomaly_status(db: AsyncSession, anomaly_id: int, is_read: bool):
        query = select(Transaction).where(
            and_(Transaction.id == anomaly_id, Transaction.is_anomaly == True)
        )
        result = await db.execute(query)
        trx = result.scalar_one_or_none()

        if not trx:
            return None

        trx.is_read = is_read
        await db.commit()
        await db.refresh(trx)
        return trx