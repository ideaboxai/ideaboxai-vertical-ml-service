import json
import pandas as pd


def predict(database_name: str, table_name: str, feature_name: str, data: pd.DataFrame):
    """
    Predict anomalies for a given feature in a specified table and database.

    Args:
        database_name (str): Name of the database.
        table_name (str): Name of the table.
        feature_name (str): Name of the feature to analyze.
        data (pd.DataFrame): DataFrame containing the data for prediction.

    Returns:
        pd.DataFrame: DataFrame with anomaly detection results.
    """
    with open("models/sandhya_aqua_erp/statistical_thresholds.json", "r") as f:
        thresholds = json.load(f)

    items = thresholds.get("data", [])
    item = next(
        (
            i
            for i in items
            if i.get("database_name") == database_name
            and i.get("table_name") == table_name
        ),
        None,
    )

    feature = next(
        (f for f in item.get("features", []) if f.get("name") == feature_name), None
    )

    # Keep only the feature_name column in the DataFrame
    data = data[[feature_name]].copy()

    for strategy in feature.get("strategies", []):
        strategy_name = strategy.get("name")
        thresholds = strategy.get("thresholds", {})
        lower_bound = thresholds.get("lower_bound")
        upper_bound = thresholds.get("upper_bound")

        data[f"{strategy_name}_lower_bound"] = lower_bound
        data[f"{strategy_name}_upper_bound"] = upper_bound

        data[f"{strategy_name}_anomaly"] = (data[feature_name] < lower_bound) | (
            data[feature_name] > upper_bound
        )

    return data
