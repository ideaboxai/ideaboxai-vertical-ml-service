from abc import ABC, abstractmethod
import pandas as pd


class BaseAnomalyDetector(ABC):
    """Abstract base class for all anomaly detection strategies."""

    def __init__(self, params: dict = None):
        self.params = params if params is not None else {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Fits the detector. For stat models, this calculates thresholds."""
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Returns predictions. -1 for anomalies, 1 for inliers."""
        pass
