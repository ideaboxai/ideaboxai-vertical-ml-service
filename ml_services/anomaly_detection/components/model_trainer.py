import joblib
import pandas as pd
from dataclasses import dataclass
from ml_services.anomaly_detection.factory import get_detector
from typing import Optional

STAT_DETECTORS = {"iqr", "z_score", "mad"}


@dataclass
class ModelTrainerConfig:
    model_name: str
    params: dict
    model_save_path: Optional[str] = None


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self, data: pd.DataFrame):
        model_params = (
            self.config.params if isinstance(self.config.params, dict) else {}
        )
        detector = get_detector(model_name=self.config.model_name, params=model_params)

        if self.config.model_name in STAT_DETECTORS:
            # Statistical detectors: fit returns thresholds
            thresholds = detector.fit(data)
            return thresholds
        else:
            # ML detectors: fit returns fitted model
            detector.fit(data)
            print(f"Saving detector to {self.config.model_save_path}")
            joblib.dump(detector, self.config.model_save_path)
