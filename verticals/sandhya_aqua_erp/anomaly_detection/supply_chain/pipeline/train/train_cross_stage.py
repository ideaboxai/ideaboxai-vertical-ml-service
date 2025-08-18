import json
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))
)

from src.anomaly_detection.components.model_trainer import (
    ModelTrainer,
    ModelTrainerConfig,
)

from verticals.sandhya_aqua_erp.data_preparation.repositories.yield_repo import (
    YieldRepository,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_yield_anomaly_detection():
    with open("verticals/sandhya_aqua_erp/configs/cross_stage_config.json") as f:
        config_data = json.load(f)

    yield_repo = YieldRepository()

    yield_data_mapper = {
        "grn_grading_yield_data": yield_repo.get_grn_grading_yield_data(),
        "soaking_yield_data": yield_repo.get_soaking_yield_data(),
        "packing_yield_data": yield_repo.get_packing_yield_data(),
    }

    yield_statistical_thresholds = []

    total_features = 0

    for item in config_data:
        cross_stage_name = item.get("cross_stage_name")

        cross_stage_stats = {"cross_stage_name": cross_stage_name, "features": []}

        dataframe = yield_data_mapper.get(cross_stage_name)
        if dataframe is None:
            logger.warning(f"No data found for cross stage: {cross_stage_name}")
            continue

        for feature in item.get("features"):
            feature_name = feature.get("name")
            strategies = feature.get("strategies", [])
            logger.info(
                f"Processing cross stage: {cross_stage_name}, feature: {feature_name}"
            )
            feature_data = dataframe[feature_name]

            feature_data = feature_data.dropna()

            feature_stats = {"feature_name": feature_name, "strategies": []}

            for strategy in strategies:
                strategy_name = strategy.get("name")
                strategy_params = strategy.get("params", {})

                strategy_stats = {
                    "name": strategy_name,
                    "params": strategy_params,
                    "thresholds": None,
                }

                logger.info(
                    f"Processing feature: {feature_name} with strategy: {strategy}"
                )

                model_trainer_config = ModelTrainerConfig(
                    model_name=strategy_name, params=strategy_params
                )
                model_trainer = ModelTrainer(model_trainer_config)

                thresholds = model_trainer.train(feature_data)

                strategy_stats["thresholds"] = thresholds
                feature_stats["strategies"].append(strategy_stats)

            cross_stage_stats["features"].append(feature_stats)

            total_features += 1
        yield_statistical_thresholds.append(cross_stage_stats)
        logger.info(
            f"Processed cross stage: {cross_stage_name} with {len(cross_stage_stats['features'])} features"
        )

    output = {
        "metadata": {
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "description": "Yield anomaly detection statistics",
            "version": "1.0.0",
            "total_cross_stages": len(yield_statistical_thresholds),
            "total_features": total_features,
        },
        "data": yield_statistical_thresholds,
    }

    output_file = "models/sandhya_aqua_erp/yield_statistical_thresholds.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)


run_yield_anomaly_detection()
