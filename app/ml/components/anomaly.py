import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class AnomalyDetectionAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_cols = [
            'withdrawal', 'amount_decimal', 'processing_lag',
            'days_since_salary', 'rolling_volatility_7d',
            'amount_global_freq', 'days_since_same_amount',
            'velocity_ratio_robust', 'is_payday_spending'
        ]

    def _calculate_single_velocity_proxy(self, vector: pd.DataFrame) -> float:
        amount = vector.iloc[0]['withdrawal']
        return amount / 2000.0

    def predict_single(self, vector: pd.DataFrame, model, scaler) -> tuple[bool, str | None]:
        if 'velocity_ratio_robust' not in vector.columns:
            vector['velocity_ratio_robust'] = self._calculate_single_velocity_proxy(vector)

        if 'is_payday_spending' not in vector.columns:
            vector['is_payday_spending'] = 0

        for col in self.feature_cols:
            if col not in vector.columns:
                vector[col] = 0.0

        X_af = vector[self.feature_cols].fillna(0)
        X_scaled = scaler.transform(X_af)
        score = model.predict(X_scaled)[0]

        is_anomaly = score == -1
        reason = None

        if is_anomaly:
            withdrawal = vector.iloc[0]['withdrawal']
            velocity = vector.iloc[0]['velocity_ratio_robust']

            reasons = []
            if withdrawal > 30000:
                reasons.append("Нетипично крупная сумма")
            if velocity > 5.0:
                reasons.append("Высокая интенсивность трат")
            if vector.iloc[0]['amount_decimal'] == 0 and withdrawal > 5000:
                reasons.append("Крупный перевод")

            if not reasons:
                reasons.append("Аномальный паттерн")

            reason = ", ".join(reasons)

        return is_anomaly, reason