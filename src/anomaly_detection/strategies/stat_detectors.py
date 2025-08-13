import numpy as np
import pandas as pd
from .base import BaseAnomalyDetector


class IQRDetector(BaseAnomalyDetector):
    def fit(self, data: pd.DataFrame):
        column = data.columns[0]
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        multiplier = self.params.get("multiplier", 1.5)
        self.lower_bound_ = q1 - multiplier * iqr
        self.upper_bound_ = q3 + multiplier * iqr
        self.is_fitted = True
        print(
            f"IQRDetector fitted. Lower: {self.lower_bound_:.2f}, Upper: {self.upper_bound_:.2f}"
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("You must call fit before predicting.")
        column = data.columns[0]
        anomalies = (data[column] < self.lower_bound_) | (
            data[column] > self.upper_bound_
        )
        return pd.Series(np.where(anomalies, -1, 1), index=data.index)


class ZScoreDetector(BaseAnomalyDetector):
    def fit(self, data: pd.DataFrame):
        column = data.columns[0]
        self.mean_ = data[column].mean()
        self.std_ = data[column].std()
        self.is_fitted = True
        print(f"ZScoreDetector fitted. Mean: {self.mean_:.2f}, Std: {self.std_:.2f}")

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("You must call fit before predicting.")
        column = data.columns[0]
        threshold = self.params.get("threshold", 3.0)
        z_scores = np.abs((data[column] - self.mean_) / self.std_)
        anomalies = z_scores > threshold
        return pd.Series(np.where(anomalies, -1, 1), index=data.index)


class MADDetector(BaseAnomalyDetector):
    def fit(self, data: pd.DataFrame):
        column = data.columns[0]
        median = data[column].median()
        mad = np.median(np.abs(data[column] - median))
        multiplier = self.params.get("multiplier", 3.0)
        self.lower_bound_ = median - multiplier * mad
        self.upper_bound_ = median + multiplier * mad
        self.is_fitted = True
        print(
            f"MADDetector fitted. Lower: {self.lower_bound_:.2f}, Upper: {self.upper_bound_:.2f}"
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("You must call fit before predicting.")
        column = data.columns[0]
        anomalies = (data[column] < self.lower_bound_) | (
            data[column] > self.upper_bound_
        )
        return pd.Series(np.where(anomalies, -1, 1), index=data.index)
