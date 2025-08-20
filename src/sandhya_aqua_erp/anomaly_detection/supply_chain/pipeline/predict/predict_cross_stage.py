import json
import pandas as pd


def predict(cross_stage_name: str, data: pd.DataFrame):
    """
    Predict anomalies for a given feature in a specified table and database.

    Args:
        cross_stage_name (str): The name of the cross stage to predict anomalies for.
        data (pd.DataFrame): The DataFrame containing the data to predict anomalies on.
    """
    with open("models/sandhya_aqua_erp/yield_statistical_thresholds.json", "r") as f:
        thresholds = json.load(f)

    items = thresholds.get("data", [])

    for item in items:
        if item.get("cross_stage_name") == cross_stage_name:
            features = item.get("features", [])
            for feature in features:
                feature_name = feature.get("feature_name")
                strategies = feature.get("strategies", [])
                if feature_name not in data.columns:
                    continue
                feature_data = data[[feature_name]].copy()

                for strategy in strategies:
                    strategy_name = strategy.get("name")
                    thresholds = strategy.get("thresholds", {})

                    lower_bound = thresholds.get("lower_bound")
                    upper_bound = thresholds.get("upper_bound")

                    feature_data[f"{feature_name}_{strategy_name}_lower_bound"] = (
                        lower_bound
                    )
                    feature_data[f"{feature_name}_{strategy_name}_upper_bound"] = (
                        upper_bound
                    )

                    feature_data[f"{feature_name}_{strategy_name}_anomaly"] = (
                        feature_data[feature_name] < lower_bound
                    ) | (feature_data[feature_name] > upper_bound)
                data = pd.concat(
                    [data, feature_data.drop(columns=[feature_name])], axis=1
                )

    return data
