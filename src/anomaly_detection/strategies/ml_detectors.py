import pandas as pd
from .base import BaseAnomalyDetector
from sklearn.ensemble import IsolationForest
# from sklearn.svm import OneClassSVM


class IsolationForestDetector(BaseAnomalyDetector):
    def fit(self, data: pd.DataFrame):
        self.model_ = IsolationForest(**self.params)
        self.model_.fit(data)
        self.is_fitted = True
        print("IsolationForestDetector fitted.")

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("You must call fit before predicting.")
        return pd.Series(self.model_.predict(data), index=data.index)
