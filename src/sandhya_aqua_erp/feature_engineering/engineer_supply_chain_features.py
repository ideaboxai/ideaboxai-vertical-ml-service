import pandas as pd


class BaseFeatureEngineer:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def engineer_features(self, table_name: str):
        raise NotImplementedError("This method should be overridden by subclasses")


class GRNFeatureEngineer(BaseFeatureEngineer):
    def engineer_features(self, table_name: str):
        return self.dataframe


class GradingFeatureEngineer(BaseFeatureEngineer):
    def engineer_features(self, table_name: str):
        return self.dataframe


class SoakingFeatureEngineer(BaseFeatureEngineer):
    def engineer_features(self, table_name: str):
        if table_name == "erpx_dev_production.pp_soaking_lot":
            return self.dataframe
        elif table_name == "erpx_dev_production.pp_soaking":
            self.dataframe["actual_soaking_time_in_minutes"] = (
                self.dataframe["closed_at"] - self.dataframe["soak_started_at"]
            ).dt.total_seconds() / 60
            return self.dataframe
        elif table_name == "erpx_dev_production.pp_soaking_readings":
            self.dataframe["average_weight_per_crate_in_kg"] = (
                self.dataframe["weight"] / self.dataframe["crates"]
            )
            return self.dataframe
        elif table_name == "erpx_dev_production.pp_soaking_temperature":
            return self.dataframe


class CookingFeatureEngineer(BaseFeatureEngineer):
    def engineer_features(self, table_name: str):
        return self.dataframe


class PackingFeatureEngineer(BaseFeatureEngineer):
    def engineer_features(self, table_name: str):
        return self.dataframe
