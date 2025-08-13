import joblib
import pandas as pd
from dataclasses import dataclass
from src.anomaly_detection.factory import get_detector


@dataclass
class ModelTrainerConfig:
    model_save_path: str
    model_name: str
    params: dict


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self, data: pd.DataFrame):
        # 1. Get the right detector strategy from the factory
        model_params = self.config.params.get(self.config.model_name, {})
        detector = get_detector(model_name=self.config.model_name, params=model_params)

        # 2. Fit the detector
        detector.fit(data)

        # 3. Save the entire fitted detector object
        print(f"Saving detector to {self.config.model_save_path}")
        joblib.dump(detector, self.config.model_save_path)
