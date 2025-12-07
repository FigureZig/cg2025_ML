import pandas as pd
import numpy as np
from datetime import timedelta


class FinancialAdvisoryEngine:
    def __init__(self, df: pd.DataFrame, predicted_categories: list):
        self.df = df.copy()
        self.df['category'] = predicted_categories
        self.df = self.df.sort_values('trx_date').reset_index(drop=True)
        self.last_date = self.df['trx_date'].max()
        self.current_balance = self.df.iloc[-1]['balance'] if not self.df.empty else 0.0

    def _build_elastic_profile(self):
        incomes = self.df[self.df['category'] == 'Salary']
        if len(incomes) >= 1:
            salary_amt = incomes['deposit'].median()
            if len(incomes) > 1:
                days_diff = incomes['trx_date'].diff().dt.days.median()
                salary_freq = int(days_diff) if not pd.isna(days_diff) and days_diff > 0 else 30
            else:
                salary_freq = 30
            last_salary = incomes['trx_date'].max()
        else:
            salary_amt = 0
            salary_freq = 30
            last_salary = self.last_date

        rents = self.df[self.df['category'] == 'Rent']
        if len(rents) > 0:
            rent_amt = rents['withdrawal'].median()
            last_rent = rents['trx_date'].max()
        else:
            rent_amt = 0
            last_rent = self.last_date

        variable_tx = self.df[
            (~self.df['category'].isin(['Salary', 'Rent'])) &
            (self.df['withdrawal'] > 0) &
            (self.df['withdrawal'] < 50000)
            ]

        total_days = max((self.last_date - self.df['trx_date'].min()).days, 1)
        cat_monthly = variable_tx.groupby('category')['withdrawal'].sum() / (total_days / 30.0)

        elasticity_rules = {
            'Food': 0.3, 'Transport': 0.5, 'Shopping': 1.0,
            'Misc': 1.0, 'Rent': 0.0, 'Salary': 0.0
        }

        profile = {}
        for cat, amount in cat_monthly.items():
            elas = elasticity_rules.get(cat, 0.8)
            profile[cat] = {
                'avg_monthly': amount,
                'min_monthly': amount * (1.0 - elas),
                'elasticity': elas
            }

        base_burn_daily = sum([v['avg_monthly'] for v in profile.values()]) / 30.0

        return {
            'salary': {'amount': salary_amt, 'freq': salary_freq, 'last_date': last_salary},
            'rent': {'amount': rent_amt, 'last_date': last_rent},
            'categories': profile,
            'burn_rates': {'comfort': base_burn_daily}
        }

    def _simulate_cashflow(self, horizon_days, profile, optimized_limits=None):
        dates = [self.last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
        balances = []
        curr = self.current_balance
        events = []

        daily_burn_map = {k: v['avg_monthly'] / 30.0 for k, v in profile['categories'].items()}
        if optimized_limits:
            for cat, limit in optimized_limits.items():
                if cat in daily_burn_map:
                    daily_burn_map[cat] = limit / 30.0

        daily_burn = sum(daily_burn_map.values())

        next_salary = profile['salary']['last_date'] + timedelta(days=profile['salary']['freq'])
        while next_salary <= self.last_date: next_salary += timedelta(days=profile['salary']['freq'])

        next_rent = profile['rent']['last_date'] + timedelta(days=30)
        while next_rent <= self.last_date: next_rent += timedelta(days=30)

        for d in dates:
            curr -= daily_burn
            if d.date() == next_salary.date():
                curr += profile['salary']['amount']
                events.append({'date': d, 'type': 'Salary', 'amount': profile['salary']['amount']})
                next_salary += timedelta(days=profile['salary']['freq'])
            if d.date() == next_rent.date():
                curr -= profile['rent']['amount']
                events.append({'date': d, 'type': 'Rent', 'amount': -profile['rent']['amount']})
                next_rent += timedelta(days=30)
            balances.append(curr)

        return pd.DataFrame({'date': dates, 'balance': balances}), events

    def generate_strategy_payload(self, goals: list = None):
        profile = self._build_elastic_profile()
        if not goals:
            horizon = 45
            target_amount = 0
        else:
            target_goal = goals[0]
            target_amount = target_goal['amount']
            goal_date = pd.to_datetime(target_goal['date'])
            horizon = (goal_date - self.last_date).days
            if horizon < 1: horizon = 30

        natural_df, natural_events = self._simulate_cashflow(horizon, profile)
        projected_balance = natural_df.iloc[-1]['balance']
        gap = target_amount - projected_balance if goals else 0

        status = 'ON_TRACK'
        if not goals: status = 'SECURE' if projected_balance > 0 else 'CRITICAL'

        strategy_steps = []
        graph_data = []
        advice = "Финансовый план стабилен."

        if gap > 0 and goals:
            needed_monthly_save = (gap / horizon) * 30.0
            sorted_cats = sorted(profile['categories'].items(), key=lambda x: x[1]['elasticity'], reverse=True)
            saved = 0
            new_limits = {}

            for cat, info in sorted_cats:
                if saved >= needed_monthly_save: break
                max_cut = info['avg_monthly'] - info['min_monthly']
                cut_needed = needed_monthly_save - saved
                actual_cut = min(max_cut, cut_needed)

                if actual_cut > 0:
                    new_limits[cat] = info['avg_monthly'] - actual_cut
                    saved += actual_cut
                    cut_percent = (actual_cut / info['avg_monthly']) * 100
                    reason = "Высокая эластичность."
                    if cat == 'Shopping': reason = "Отказ от спонтанных покупок"
                    if cat == 'Food' and cut_percent > 10: reason = "Готовим дома чаще"

                    strategy_steps.append({
                        'category': cat,
                        'current_monthly': round(info['avg_monthly'], 0),
                        'cut_monthly': round(actual_cut, 0),
                        'cut_percent': round(cut_percent, 0),
                        'new_limit_monthly': round(info['avg_monthly'] - actual_cut, 0),
                        'reason': reason
                    })

            if saved < needed_monthly_save * 0.95:
                status = 'UNREALISTIC_DATE'
                max_save_potential = (profile['burn_rates']['comfort'] - sum(
                    [v['min_monthly'] for v in profile['categories'].values()]) / 30.0) * 30.0
                natural_surplus = (projected_balance / horizon) * 30.0
                total_power = max_save_potential + natural_surplus

                suggested_date = None
                if total_power > 0:
                    months_needed = target_goal['amount'] / total_power
                    suggested_date = (self.last_date + timedelta(days=int(months_needed * 30))).strftime('%Y-%m-%d')

                advice = f"Цель недостижима в срок. Реальная дата: {suggested_date}"
            else:
                status = 'ACHIEVABLE_WITH_CUTS'
                advice = f"Сократите расходы на {int(saved)} у.е./мес."

            opt_df, _ = self._simulate_cashflow(horizon, profile, optimized_limits=new_limits)

            for i in range(len(natural_df)):
                graph_data.append({
                    'date': natural_df.iloc[i]['date'].strftime('%Y-%m-%d'),
                    'balance': float(round(natural_df.iloc[i]['balance'], 2)),
                    'balance_optimized': float(round(opt_df.iloc[i]['balance'], 2))
                })
        else:
            for i in range(len(natural_df)):
                graph_data.append({
                    'date': natural_df.iloc[i]['date'].strftime('%Y-%m-%d'),
                    'balance': float(round(natural_df.iloc[i]['balance'], 2)),
                    'balance_optimized': None
                })

        return {
            'meta': {
                'status': status,
                'goal_gap': float(round(gap, 2)),
                'current_balance': float(round(self.current_balance, 2))
            },
            'advice': {
                'message': advice,
                'steps': strategy_steps
            },
            'simulation': {
                'graph': graph_data,
                'events': [{'date': e['date'].strftime('%Y-%m-%d'), 'type': e['type'], 'amount': float(e['amount'])} for
                           e in natural_events]
            }
        }