from .strategies.base import BaseAnomalyDetector
from .strategies.stat_detectors import IQRDetector, ZScoreDetector
from .strategies.ml_detectors import IsolationForestDetector


def get_detector(model_name: str, params: dict = None) -> BaseAnomalyDetector:
    """Factory function to get an anomaly detector instance."""
    detectors = {
        "iqr": IQRDetector,
        "z_score": ZScoreDetector,
        "isolation_forest": IsolationForestDetector,
        # "one_class_svm": OneClassSVMDetector, # Add this when implemented
    }
    detector_class = detectors.get(model_name)
    if not detector_class:
        raise ValueError(f"Detector '{model_name}' not recognized.")

    return detector_class(params=params)
