import pandas as pd
from catboost import CatBoostClassifier
from app.ml.components.feature_eng import FeatureExtractionPipeline
from app.ml.components.advisory import FinancialAdvisoryEngine
from app.ml.components.anomaly import AnomalyDetectionAgent
from app.ml.loader import MLLoader
from app.config import settings


class MLEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLEngine, cls).__new__(cls)
        return cls._instance

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        if 'trx_date' not in df.columns:
            df['trx_date'] = pd.Timestamp.now()
        else:
            df['trx_date'] = pd.to_datetime(df['trx_date'])

        if 'type' in df.columns and 'amount' in df.columns:
            if 'withdrawal' not in df.columns:
                df['withdrawal'] = df.apply(lambda x: x['amount'] if x['type'] == 'withdrawal' else 0, axis=1)
            if 'deposit' not in df.columns:
                df['deposit'] = df.apply(lambda x: x['amount'] if x['type'] == 'deposit' else 0, axis=1)

        if 'balance_after' in df.columns and 'balance' not in df.columns:
            df['balance'] = df['balance_after']

        return df

    def predict_transaction(self, trx_data: dict) -> dict:
        df = pd.DataFrame([trx_data])
        df = self._prepare_df(df)

        result = {"category": "Misc", "is_anomaly": False, "anomaly_reason": None}

        try:
            pipeline = FeatureExtractionPipeline(df)
            X = pipeline.get_features_for_model()
        except Exception as e:
            print(f"ML Pipeline Error: {e}")
            return result

        classifier = MLLoader.get_classifier()
        if classifier:
            try:
                preds = classifier.predict(X).flatten()
                result["category"] = str(preds[0]).replace("['", "").replace("']", "")
            except Exception:
                pass

        anom_model, scaler = MLLoader.get_anomaly_artifacts()
        if anom_model and scaler:
            try:
                agent = AnomalyDetectionAgent(df)
                is_anom, reason = agent.predict_single(X, anom_model, scaler)
                result["is_anomaly"] = bool(is_anom)
                result["anomaly_reason"] = reason
            except Exception:
                pass

        return result

    def generate_strategy(self, history_df: pd.DataFrame, goals: list = None, current_date=None) -> dict:
        try:
            if history_df.empty:
                raise ValueError("No data history")

            history_df = self._prepare_df(history_df)
            pipeline = FeatureExtractionPipeline(history_df)
            X = pipeline.get_features_for_model()

            classifier = MLLoader.get_classifier()
            if classifier:
                try:
                    preds = classifier.predict(X).flatten()
                    preds = [str(p).replace("['", "").replace("']", "") for p in preds]
                except Exception:
                    preds = ['Misc'] * len(history_df)
            else:
                preds = ['Misc'] * len(history_df)

            engine = FinancialAdvisoryEngine(history_df, preds)

            sim_start = current_date if current_date else history_df['trx_date'].max()
            engine.last_date = sim_start

            if current_date:
                mask = history_df['trx_date'] <= current_date
                if mask.any():
                    engine.current_balance = history_df.loc[mask, 'balance'].iloc[-1]

            if goals:
                t_date = pd.to_datetime(goals[0]['date'])
                if t_date <= sim_start:
                    return {
                        "meta": {"status": "ERROR", "goal_gap": 0},
                        "advice": {"message": "Дата цели должна быть в будущем!", "steps": []},
                        "simulation": {"events": [], "graph": []}
                    }

            return engine.generate_strategy_payload(goals)
        except Exception as e:
            return {
                "meta": {"status": "CRITICAL_ERROR", "goal_gap": 0},
                "advice": {"message": f"Ошибка расчета: {str(e)}", "steps": []},
                "simulation": {"events": [], "graph": []}
            }

    def retrain_model(self, full_df: pd.DataFrame) -> dict:
        try:
            full_df = self._prepare_df(full_df)
            pipeline = FeatureExtractionPipeline(full_df)
            X = pipeline.get_features_for_model()
            y = full_df['category']

            model = CatBoostClassifier(iterations=100, depth=4, verbose=False)
            model.fit(X, y)
            model.save_model(str(settings.MODEL_PATH))
            MLLoader.classifier = model

            return {"status": "success", "samples": len(full_df)}
        except Exception as e:
            return {"status": "error", "reason": str(e)}


ml_engine = MLEngine()