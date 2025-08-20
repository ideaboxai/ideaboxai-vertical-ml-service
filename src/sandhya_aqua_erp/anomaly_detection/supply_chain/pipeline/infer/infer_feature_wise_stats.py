"""
Feature-wise anomaly detection for Sandhya Aqua ERP supply chain data.

This module processes feature data from various tables and applies statistical
anomaly detection strategies (IQR, MAD) to generate thresholds for anomaly detection.
"""

import json
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))
)

from ml_services.anomaly_detection.components.model_trainer import (
    ModelTrainer,
    ModelTrainerConfig,
)

from src.sandhya_aqua_erp.repositories.cooking_repo import (
    CookingRepository,
)
from src.sandhya_aqua_erp.repositories.grading_repo import (
    GradingRepository,
)
from src.sandhya_aqua_erp.repositories.grn_repo import (
    GRNRepository,
)
from src.sandhya_aqua_erp.repositories.packing_repo import (
    PackingRepository,
)
from src.sandhya_aqua_erp.repositories.soaking_repo import (
    SoakingRepository,
)

from src.sandhya_aqua_erp.feature_engineering.engineer_supply_chain_features import (
    GRNFeatureEngineer,
    GradingFeatureEngineer,
    SoakingFeatureEngineer,
    CookingFeatureEngineer,
    PackingFeatureEngineer,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_feature_wise_anomaly_detection():
    """
    Main function to run feature-wise anomaly detection.
    It reads configuration, fetches data from repositories, applies statistical
    anomaly detection strategies, and saves the thresholds to a JSON file.
    """

    with open(
        "src/sandhya_aqua_erp/anomaly_detection/supply_chain/configs/feature_wise_config.json",
        "r",
    ) as file:
        config_data = json.load(file)

        grn_repo = GRNRepository()
        grading_repo = GradingRepository()
        soaking_repo = SoakingRepository()
        cooking_repo = CookingRepository()
        packing_repo = PackingRepository()

        repo_mapper = {
            "GRNRepository": grn_repo,
            "GradingRepository": grading_repo,
            "SoakingRepository": soaking_repo,
            "CookingRepository": cooking_repo,
            "PackingRepository": packing_repo,
        }

        statistical_thresholds = []

        total_tables = 0
        total_features = 0

        # Using the loaded configuration data, fetch the data from the repositories
        # Iterating through each table
        for item in config_data:
            repo_to_use = item.get("repository_class")
            database_name = item.get("database_name")
            table_name = item.get("table_name")

            table_stats = {
                "database_name": database_name,
                "table_name": table_name,
                "features": [],
            }

            repo_instance = repo_mapper.get(repo_to_use)

            # Step 1: Fetch data from the repository

            if repo_instance:
                logger.info(f"Fetching data from {repo_to_use} for table {table_name}")
                dataframe = repo_instance.get_individual_table(table_name)

                if dataframe is not None:
                    logger.info(
                        f"Successfully fetched data from {repo_to_use} for table {table_name}"
                    )
                else:
                    logger.warning(
                        f"Failed to fetch data from {repo_to_use} for table {table_name}"
                    )

            # Step 2: Perform feature engineering
            feature_engineer = None
            if repo_to_use == "GRNRepository":
                feature_engineer = GRNFeatureEngineer(dataframe)
            elif repo_to_use == "GradingRepository":
                feature_engineer = GradingFeatureEngineer(dataframe)
            elif repo_to_use == "SoakingRepository":
                feature_engineer = SoakingFeatureEngineer(dataframe)
            elif repo_to_use == "CookingRepository":
                feature_engineer = CookingFeatureEngineer(dataframe)
            elif repo_to_use == "PackingRepository":
                feature_engineer = PackingFeatureEngineer(dataframe)

            if feature_engineer:
                dataframe = feature_engineer.engineer_features(
                    table_name=f"{database_name}.{table_name}"
                )
                logger.info(f"Feature engineering completed for {table_name}")
            else:
                logger.warning(f"No feature engineer found for {repo_to_use}")

            # Iterating through each feature
            for feature in item.get("features"):
                feature_name = feature.get("name")

                feature_stats = {"name": feature_name, "strategies": []}

                logger.info(
                    f"Processing feature: {feature.get('name')} of table {table_name}"
                )
                feature_data = None

                if feature_name and dataframe is not None:
                    feature_data = dataframe[feature_name]

                # Iterating through each strategy
                for strategy in feature.get("strategies") or []:
                    strategy_name = strategy.get("name")

                    strategy_stats = {
                        "name": strategy_name,
                        "params": strategy.get("params", {}),
                        "thresholds": None,
                    }

                    logger.info(
                        f"Applying strategy: {strategy.get('name')} on feature: {feature_name} of table {table_name}"
                    )
                    params = strategy.get("params", {})

                    model_trainer_config = ModelTrainerConfig(
                        model_name=strategy_name, params=params
                    )

                    model_trainer = ModelTrainer(model_trainer_config)

                    thresholds = model_trainer.train(feature_data)

                    strategy_stats["thresholds"] = thresholds
                    feature_stats["strategies"].append(strategy_stats)

                # Append the feature statistics to the table statistics
                table_stats["features"].append(feature_stats)
                total_features += 1

            # Append the table statistics to the overall statistical thresholds
            statistical_thresholds.append(table_stats)
            total_tables += 1

        # Add metadata
        output = {
            "metadata": {
                "generated_at": __import__("datetime").datetime.now().isoformat(),
                "description": "Feature statistics for anomaly detection",
                "version": "1.0.0",
                "total_tables": total_tables,
                "total_features": total_features,
            },
            "data": statistical_thresholds,
        }

        # Save the statistical thresholds to a JSON file
        output_path = "models/sandhya_aqua_erp/statistical_thresholds.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as outfile:
            json.dump(output, outfile, indent=4)


run_feature_wise_anomaly_detection()
