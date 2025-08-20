from abc import ABC, abstractmethod


class BaseAnomalyDetector(ABC):
    """Abstract base class for all anomaly detection strategies."""

    def __init__(self, params: dict = None):
        self.params = params if params is not None else {}

    @abstractmethod
    def fit(self):
        """Fits the detector. For stat models, this calculates thresholds."""
        pass

    @abstractmethod
    def predict(self):
        """Returns predictions."""
        pass
