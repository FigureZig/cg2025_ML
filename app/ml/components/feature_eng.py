import pandas as pd
import numpy as np


class FeatureExtractionPipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        self.df['trx_date'] = pd.to_datetime(self.df['trx_date'], errors='coerce')
        if 'posting_date' not in self.df.columns:
            self.df['posting_date'] = self.df['trx_date']
        else:
            self.df['posting_date'] = pd.to_datetime(self.df['posting_date'], errors='coerce')

        self.df = self.df.dropna(subset=['trx_date']).reset_index(drop=True)

        if not pd.api.types.is_datetime64_any_dtype(self.df['trx_date']):
            self.df['trx_date'] = self.df['trx_date'].astype('datetime64[ns]')

        if not self.df.empty:
            self._run_pipeline()

    def _run_pipeline(self):
        self._encode_cyclical_time()
        self._engineer_transactional_dna()
        self._engineer_contextual_state()
        self._engineer_consistency_metrics()
        self._engineer_volatility_metrics()

    def _encode_cyclical_time(self):
        safe_posting = self.df['posting_date'].fillna(self.df['trx_date'])
        self.df['processing_lag'] = (safe_posting - self.df['trx_date']).dt.days.fillna(0)

        self.df['day_sin'] = np.sin(2 * np.pi * self.df['trx_date'].dt.day / 31.0)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['trx_date'].dt.day / 31.0)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['trx_date'].dt.month / 12.0)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['trx_date'].dt.month / 12.0)
        self.df['weekday_sin'] = np.sin(2 * np.pi * self.df['trx_date'].dt.dayofweek / 7.0)
        self.df['weekday_cos'] = np.cos(2 * np.pi * self.df['trx_date'].dt.dayofweek / 7.0)

    def _engineer_transactional_dna(self):
        self.df['amount_decimal'] = self.df['withdrawal'] % 1
        self.df['is_round_100'] = (self.df['withdrawal'] % 100 == 0).astype(int)
        self.df['is_round_500'] = (self.df['withdrawal'] % 500 == 0).astype(int)
        self.df['log_amount'] = np.log1p(self.df['withdrawal'])

        expanding_mean = self.df['withdrawal'].expanding().mean()
        expanding_std = self.df['withdrawal'].expanding().std().fillna(1)
        self.df['amount_z_score'] = (self.df['withdrawal'] - expanding_mean) / expanding_std

    def _engineer_contextual_state(self):
        self.df['is_salary_ref'] = self.df['ref_no'].astype(str).str.contains(r'^CHAS[A-Z]?\d*', regex=True,
                                                                              na=False).astype(int)

        self.df['last_salary_date'] = pd.NaT
        salary_dates = self.df.loc[self.df['is_salary_ref'] == 1, 'trx_date']

        if not salary_dates.empty:
            self.df.loc[salary_dates.index, 'last_salary_date'] = salary_dates
            self.df['last_salary_date'] = self.df['last_salary_date'].ffill()
            self.df['last_salary_date'] = pd.to_datetime(self.df['last_salary_date'], errors='coerce')

            delta = self.df['trx_date'] - self.df['last_salary_date']
            self.df['days_since_salary'] = delta.dt.days.fillna(-1)
        else:
            self.df['days_since_salary'] = -1

        if 'last_salary_date' in self.df.columns:
            self.df.drop(columns=['last_salary_date'], inplace=True)

    def _engineer_consistency_metrics(self):
        amount_counts = self.df['withdrawal'].value_counts()
        self.df['amount_global_freq'] = self.df['withdrawal'].map(amount_counts)

        self.df['days_since_same_amount'] = (
            self.df.groupby('withdrawal')['trx_date']
            .diff()
            .dt.days
            .fillna(-1)
        )

    def _engineer_volatility_metrics(self):
        self.df['net_flow'] = self.df['deposit'] - self.df['withdrawal']
        daily_stats = self.df.groupby('trx_date')['net_flow'].sum().rolling('7D').std().fillna(0)
        self.df['rolling_volatility_7d'] = self.df['trx_date'].map(daily_stats).fillna(0)

    def get_features_for_model(self):
        feature_cols = [
            'withdrawal', 'deposit', 'processing_lag',
            'day_sin', 'day_cos', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
            'amount_decimal', 'is_round_100', 'is_round_500', 'log_amount', 'amount_z_score',
            'is_salary_ref', 'days_since_salary',
            'amount_global_freq', 'days_since_same_amount',
            'rolling_volatility_7d'
        ]

        for col in feature_cols:
            if col not in self.df.columns:
                self.df[col] = 0.0

        return self.df[feature_cols].fillna(0)