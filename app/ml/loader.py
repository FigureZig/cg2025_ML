import joblib
import os
from catboost import CatBoostClassifier
from app.config import settings


class MLLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLLoader, cls).__new__(cls)
            cls.classifier = None
            cls.anomaly_model = None
            cls.scaler = None
        return cls._instance

    @classmethod
    def load(cls):
        if os.path.exists(settings.MODEL_PATH):
            cls.classifier = CatBoostClassifier()
            cls.classifier.load_model(str(settings.MODEL_PATH))

        if os.path.exists(settings.ANOMALY_MODEL_PATH):
            cls.anomaly_model = joblib.load(settings.ANOMALY_MODEL_PATH)
            cls.scaler = joblib.load(settings.SCALER_PATH)

    @classmethod
    def get_classifier(cls):
        return cls.classifier

    @classmethod
    def get_anomaly_artifacts(cls):
        return cls.anomaly_model, cls.scaler