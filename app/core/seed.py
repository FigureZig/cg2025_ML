import pandas as pd
import os
import joblib
import numpy as np
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.transaction import Transaction
from app.config import settings
from app.ml.components.feature_eng import FeatureExtractionPipeline


# --- HELPER FOR ANOMALY DETECTION ---
class BatchAnomalyLabeler:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def apply_detection(self, model, scaler):
        daily_sum = self.df.groupby('trx_date')['withdrawal'].sum()
        idx = pd.date_range(daily_sum.index.min(), daily_sum.index.max())
        daily_sum = daily_sum.reindex(idx, fill_value=0)

        velocity_3d = daily_sum.rolling('3D').sum()

        global_median = daily_sum[daily_sum > 0].median()
        soft_floor = max(3000, global_median) if not pd.isna(global_median) else 3000

        baseline_30d = daily_sum.rolling('30D').mean().fillna(soft_floor)
        baseline = np.maximum(baseline_30d, soft_floor)

        self.df['velocity_3d'] = self.df['trx_date'].map(velocity_3d).fillna(0)
        self.df['baseline'] = self.df['trx_date'].map(baseline).fillna(soft_floor)
        self.df['velocity_ratio_robust'] = self.df['velocity_3d'] / self.df['baseline']

        # Упрощенная логика Payday для истории (раз в 30 дней пик депозита)
        self.df['is_payday_spending'] = 0

        # Pipeline фичей
        pipeline = FeatureExtractionPipeline(self.df)
        df_features = pipeline.get_features_for_model()

        df_features['velocity_ratio_robust'] = self.df['velocity_ratio_robust']
        df_features['is_payday_spending'] = self.df['is_payday_spending']

        # Колонки строго как при обучении
        feature_cols = [
            'withdrawal', 'amount_decimal', 'processing_lag',
            'days_since_salary', 'rolling_volatility_7d',
            'velocity_ratio_robust', 'is_payday_spending'
        ]

        for col in feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0.0

        X = df_features[feature_cols].fillna(0)

        # 1. ML Detection
        if model and scaler:
            try:
                X_scaled = scaler.transform(X)
                scores = model.predict(X_scaled)
                self.df['is_anomaly'] = (scores == -1)
            except:
                self.df['is_anomaly'] = False
        else:
            self.df['is_anomaly'] = False

        # 2. Hard Rules (Добиваем эвристикой, чтобы точно было красиво)
        # Если сумма > 15к для Misc/Food - помечаем аномалией принудительно
        mask_large = (self.df['withdrawal'] > 15000) & (self.df['category'].isin(['Misc', 'Food', 'Transport']))
        self.df.loc[mask_large, 'is_anomaly'] = True

        # 3. Generate Reasons (РУССКИЕ ТЕГИ ДЛЯ ГЕНЕРАТОРА)
        reasons = []
        for idx, row in self.df.iterrows():
            if not row['is_anomaly']:
                reasons.append(None)
                continue

            r_list = []

            # Ловим Velocity
            if row['velocity_ratio_robust'] > 2.0:
                r_list.append(f"высокая интенсивность x{row['velocity_ratio_robust']:.1f}")

            # Ловим крупные суммы (снизил порог до 10к)
            if row['withdrawal'] > 10000:
                r_list.append("значительная сумма")

            # Если ничего не подошло, но аномалия
            if not r_list and row['withdrawal'] > 5000:
                r_list.append("новая крупная покупка")

            reasons.append(", ".join(r_list) if r_list else "нетипичный паттерн")

        self.df['anomaly_reason'] = reasons
        return self.df


# --- MAIN SEED FUNCTION ---

def custom_date_parser(date_str):
    if pd.isna(date_str): return pd.NaT
    date_str = str(date_str).strip()
    for sep in ['/', '.', '-']:
        if sep in date_str:
            parts = date_str.split(sep)
            if len(parts) == 3:
                d, m, y = parts
                if len(y) == 2: y = '20' + y
                return pd.Timestamp(year=int(y), month=int(m), day=int(d))
    return pd.NaT


async def seed_data(db: AsyncSession):
    result = await db.execute(select(func.count(Transaction.id)))
    count = result.scalar()
    if count > 0:
        print(f"Database already has {count} transactions. Skipping seed.")
        return

    csv_path = "data/ci_data.csv"
    if not os.path.exists(csv_path): return

    try:
        print(f"Reading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)

        col_map = {
            'Date': 'trx_date', 'Category': 'category', 'RefNo': 'ref_no',
            'Withdrawal': 'withdrawal', 'Deposit': 'deposit', 'Balance': 'balance_after'
        }
        df = df.rename(columns=col_map)
        df['trx_date'] = df['trx_date'].apply(custom_date_parser)
        df = df.dropna(subset=['trx_date']).sort_values('trx_date').reset_index(drop=True)

        print("Running ML tagging...")
        model_path = settings.ANOMALY_MODEL_PATH
        scaler_path = settings.SCALER_PATH

        anom_model = None
        scaler = None

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                anom_model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                labeler = BatchAnomalyLabeler(df)
                df = labeler.apply_detection(anom_model, scaler)
                print(f"Detected {df['is_anomaly'].sum()} anomalies.")
            except Exception as e:
                print(f"ML tagging error: {e}")
                df['is_anomaly'] = False
                df['anomaly_reason'] = None
        else:
            df['is_anomaly'] = False
            df['anomaly_reason'] = None

        transactions_to_add = []
        for _, row in df.iterrows():
            try:
                w = float(row.get('withdrawal', 0) if pd.notna(row.get('withdrawal')) else 0)
                d = float(row.get('deposit', 0) if pd.notna(row.get('deposit')) else 0)
                bal = float(row.get('balance_after', 0) if pd.notna(row.get('balance_after')) else 0)
            except:
                continue

            if w > 0:
                amt, typ = w, 'withdrawal'
            elif d > 0:
                amt, typ = d, 'deposit'
            else:
                continue

            cat = row.get('category', 'Misc')
            if pd.isna(cat) or str(cat).strip() == '': cat = 'Misc'

            transactions_to_add.append(Transaction(
                trx_date=row['trx_date'],
                amount=amt,
                type=typ,
                category=str(cat).strip(),
                ref_no=str(row.get('ref_no', '')),
                balance_after=bal,
                is_anomaly=bool(row.get('is_anomaly', False)),
                anomaly_reason=str(row.get('anomaly_reason', '')) if row.get('anomaly_reason') else None,
                is_read=False
            ))

        if transactions_to_add:
            db.add_all(transactions_to_add)
            await db.commit()
            print(f"Seeded {len(transactions_to_add)} transactions.")

    except Exception as e:
        print(f"Seeding failed: {e}")