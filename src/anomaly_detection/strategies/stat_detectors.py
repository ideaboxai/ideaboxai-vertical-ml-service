import numpy as np
import pandas as pd
from .base import BaseAnomalyDetector


class IQRDetector(BaseAnomalyDetector):
    """
    Interquartile Range (IQR) based anomaly detection.
    This detector calculates the IQR and flags anomalies as those that fall
    outside a specified multiplier of the IQR.
    """

    def fit(self, data: pd.DataFrame):
        data = data.dropna()

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        multiplier = self.params.get("iqr_multiplier", 1.5)

        # Features can be non-negative or percentages
        is_non_negative_feature = self.params.get("is_non_negative_feature", True)

        if is_non_negative_feature:
            lower_bound = max(0, q1 - multiplier * iqr)
        else:
            lower_bound = q1 - multiplier * iqr

        upper_bound = q3 + multiplier * iqr

        print(f"IQRDetector fitted. Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f}")

        thresholds = {"lower_bound": lower_bound, "upper_bound": upper_bound}

        return thresholds

    def predict(self, data: pd.DataFrame) -> pd.Series:
        pass


class ZScoreDetector(BaseAnomalyDetector):
    """
    Z-Score based anomaly detection.
    This detector calculates the Z-Score for each data point and flags anomalies
    as those that exceed a specified threshold (default is 3.0).
    """

    def fit(self, data: pd.DataFrame):
        data = data.dropna()
        column = data.columns[0]
        mean_ = data[column].mean()
        std_ = data[column].std()
        z_max_threshold = self.params.get("z_max_threshold", 3.0)

        print(
            f"ZScoreDetector fitted. Mean: {mean_:.2f}, Std: {std_:.2f}, Z-Max Threshold: {z_max_threshold:.2f}"
        )

        lower_bound = mean_ - z_max_threshold * std_
        upper_bound = mean_ + z_max_threshold * std_

        thresholds = {"lower_bound": lower_bound, "upper_bound": upper_bound}

        return thresholds

    def predict(self, data: pd.DataFrame) -> pd.Series:
        pass


class MADDetector(BaseAnomalyDetector):
    """
    Median Absolute Deviation (MAD) based anomaly detection.
    This detector calculates the median and MAD for the data and flags anomalies
    as those that exceed a specified multiplier of the MAD.
    """

    def fit(self, data: pd.DataFrame):
        data = data.dropna()
        # Handle both DataFrame and Series input
        if isinstance(data, pd.DataFrame):
            column = data.columns[0]
            values = data[column]
        else:
            values = data

        median = values.median()
        mad = np.median(np.abs(values - median))
        multiplier = self.params.get("mad_multiplier", 3.5)

        lower_bound = median - multiplier * mad
        upper_bound = median + multiplier * mad

        thresholds = {"lower_bound": lower_bound, "upper_bound": upper_bound}
        return thresholds

    def predict(self, data: pd.DataFrame) -> pd.Series:
        pass
