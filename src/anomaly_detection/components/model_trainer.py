import joblib
import pandas as pd
from dataclasses import dataclass
from src.anomaly_detection.factory import get_detector

STAT_DETECTORS = {"iqr", "z_score", "mad"}


@dataclass
class ModelTrainerConfig:
    model_save_path: str
    model_name: str
    params: dict


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self, data: pd.DataFrame):
        model_params = self.config.params.get(self.config.model_name, {})
        detector = get_detector(model_name=self.config.model_name, params=model_params)

        if self.config.model_name in STAT_DETECTORS:
            # Statistical detectors: fit returns thresholds
            thresholds = detector.fit(data)
            print(f"Saving thresholds to {self.config.model_save_path}")
            joblib.dump(thresholds, self.config.model_save_path)
        else:
            # ML detectors: fit returns fitted model
            detector.fit(data)
            print(f"Saving detector to {self.config.model_save_path}")
            joblib.dump(detector, self.config.model_save_path)
