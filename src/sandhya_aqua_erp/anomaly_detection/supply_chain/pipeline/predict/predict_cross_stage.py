# import json
# import pandas as pd


# def predict(cross_stage_name: str, data: pd.DataFrame):
#     """
#     Predict anomalies for a given feature in a specified table and database.

#     Args:
#         cross_stage_name (str): The name of the cross stage to predict anomalies for.
#         data (pd.DataFrame): The DataFrame containing the data to predict anomalies on.
#     """
#     with open("models/sandhya_aqua_erp/yield_statistical_thresholds.json", "r") as f:
#         thresholds = json.load(f)

#     items = thresholds.get("data", [])

#     for item in items:
#         if item.get("cross_stage_name") == cross_stage_name:
#             features = item.get("features", [])
#             for feature in features:
#                 feature_name = feature.get("feature_name")
#                 strategies = feature.get("strategies", [])
#                 if feature_name not in data.columns:
#                     continue
#                 feature_data = data[[feature_name]].copy()

#                 for strategy in strategies:
#                     strategy_name = strategy.get("name")
#                     thresholds = strategy.get("thresholds", {})

#                     lower_bound = thresholds.get("lower_bound")
#                     upper_bound = thresholds.get("upper_bound")

#                     feature_data[f"{feature_name}_{strategy_name}_lower_bound"] = (
#                         lower_bound
#                     )
#                     feature_data[f"{feature_name}_{strategy_name}_upper_bound"] = (
#                         upper_bound
#                     )
#                     feature_data[f"{feature_name}_{strategy_name}_anomaly"] = (
#                         feature_data[feature_name] < lower_bound
#                     ) | (feature_data[feature_name] > upper_bound)
#                 data = pd.concat(
#                     [data, feature_data.drop(columns=[feature_name])], axis=1
#                 )

#     return data


import json
import pandas as pd
import numpy as np


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

                # Track anomaly flags for each strategy
                anomaly_flags = []
                strategy_results = {}

                for strategy in strategies:
                    strategy_name = strategy.get("name")
                    thresholds_dict = strategy.get("thresholds", {})

                    lower_bound = thresholds_dict.get("lower_bound")
                    upper_bound = thresholds_dict.get("upper_bound")

                    feature_data[f"{feature_name}_{strategy_name}_lower_bound"] = (
                        lower_bound
                    )
                    feature_data[f"{feature_name}_{strategy_name}_upper_bound"] = (
                        upper_bound
                    )

                    # Individual strategy anomaly detection
                    strategy_anomaly = (feature_data[feature_name] < lower_bound) | (
                        feature_data[feature_name] > upper_bound
                    )

                    feature_data[f"{feature_name}_{strategy_name}_anomaly"] = (
                        strategy_anomaly
                    )
                    anomaly_flags.append(strategy_anomaly)

                    # Store strategy results for remarks generation
                    strategy_results[strategy_name] = {
                        "anomaly": strategy_anomaly,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                    }

                # Aggressive Approach: Flag as anomaly only if ALL strategies suggest anomaly
                if anomaly_flags:
                    all_strategies_anomaly = pd.concat(anomaly_flags, axis=1).all(
                        axis=1
                    )
                    feature_data[f"{feature_name}_final_anomaly"] = (
                        all_strategies_anomaly
                    )

                    # Calculate deviation for anomalies
                    feature_data[f"{feature_name}_deviation"] = np.where(
                        all_strategies_anomaly,
                        _calculate_deviation(
                            feature_data[feature_name], strategy_results
                        ),
                        0.0,
                    )

                    # Generate remarks
                    feature_data[f"{feature_name}_remarks"] = _generate_remarks(
                        feature_data[feature_name], strategy_results, feature_name
                    )

                data = pd.concat(
                    [data, feature_data.drop(columns=[feature_name])], axis=1
                )

    return data


def _calculate_deviation(values, strategy_results):
    """
    Calculate deviation as percentage from the nearest threshold.

    Args:
        values (pd.Series): Feature values
        strategy_results (dict): Dictionary containing strategy results

    Returns:
        pd.Series: Deviation percentages
    """
    deviations = []

    for value in values:
        min_deviation = float("inf")

        for results in strategy_results.values():
            lower_bound = results["lower_bound"]
            upper_bound = results["upper_bound"]

            if (
                lower_bound is not None
                and pd.notna(lower_bound)
                and value is not None
                and pd.notna(value)
                and value < lower_bound
            ):
                deviation = abs((value - lower_bound) / lower_bound * 100)
                min_deviation = min(min_deviation, deviation)
            elif (
                upper_bound is not None
                and pd.notna(upper_bound)
                and value is not None
                and pd.notna(value)
                and value > upper_bound
            ):
                deviation = abs((value - upper_bound) / upper_bound * 100)
                min_deviation = min(min_deviation, deviation)

        deviations.append(min_deviation if min_deviation != float("inf") else 0.0)

    return pd.Series(deviations)


def _generate_remarks(values, strategy_results, feature_name):
    """
    Generate descriptive remarks for each value based on anomaly detection results.

    Args:
        values (pd.Series): Feature values
        strategy_results (dict): Dictionary containing strategy results
        feature_name (str): Name of the feature

    Returns:
        pd.Series: Remarks for each value
    """
    remarks = []

    # Extract feature type for better remarks
    feature_type = feature_name.lower()

    for i, value in enumerate(values):
        value_remarks = []

        for strategy_name, results in strategy_results.items():
            if results["anomaly"].iloc[i]:
                lower_bound = results["lower_bound"]
                upper_bound = results["upper_bound"]

                if pd.notna(lower_bound) and value < lower_bound:
                    # Value is below lower bound
                    deviation_pct = abs((value - lower_bound) / lower_bound * 100)
                    if deviation_pct > 80:
                        intensity = "very low"
                    elif deviation_pct > 10:
                        intensity = "low"
                    else:
                        intensity = "slightly low"

                    value_remarks.append(
                        f"{intensity} {_get_feature_descriptor(feature_type)}"
                    )

                elif pd.notna(upper_bound) and value > upper_bound:
                    # Value is above upper bound
                    deviation_pct = abs((value - upper_bound) / upper_bound * 100)
                    if deviation_pct > 80:
                        intensity = "very high"
                    elif deviation_pct > 10:
                        intensity = "high"
                    else:
                        intensity = "slightly high"

                    value_remarks.append(
                        f"{intensity} {_get_feature_descriptor(feature_type)}"
                    )

        if value_remarks:
            remarks.append("; ".join(set(value_remarks)))  # Remove duplicates
        else:
            remarks.append("normal")

    return pd.Series(remarks)


def _get_feature_descriptor(feature_type):
    """
    Get appropriate descriptor based on feature type.

    Args:
        feature_type (str): Feature name in lowercase

    Returns:
        str: Appropriate descriptor
    """
    if "weight" in feature_type:
        return "weight"
    elif "time" in feature_type or "duration" in feature_type:
        return "time"
    elif "count" in feature_type or "quantity" in feature_type:
        return "count"
    elif "temperature" in feature_type or "temp" in feature_type:
        return "temperature"
    elif "pressure" in feature_type:
        return "pressure"
    elif "rate" in feature_type:
        return "rate"
    else:
        return "value"
