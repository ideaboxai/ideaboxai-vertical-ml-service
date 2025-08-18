import json
import sys
import os


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from src.anomaly_detection.components.model_trainer import (
    ModelTrainer,
    ModelTrainerConfig,
)

from verticals.sandhya_aqua_erp.data_preparation.repositories.cooking_repo import (
    CookingRepository,
)
from verticals.sandhya_aqua_erp.data_preparation.repositories.grading_repo import (
    GradingRepository,
)
from verticals.sandhya_aqua_erp.data_preparation.repositories.grn_repo import (
    GRNRepository,
)
from verticals.sandhya_aqua_erp.data_preparation.repositories.packing_repo import (
    PackingRepository,
)
from verticals.sandhya_aqua_erp.data_preparation.repositories.soaking_repo import (
    SoakingRepository,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_stage_wise_anomaly_detection():
    """
    Main function to run stage-wise anomaly detection.
    It reads configuration, fetches data from repositories, applies statistical
    anomaly detection strategies, and saves the thresholds to a JSON file.
    """

    with open("verticals/sandhya_aqua_erp/configs/stage_wise_config.json", "r") as file:
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

        for config in config_data:
            repository_class = config["repository_class"]
            strategies = config["strategies"]
            features = config.get("features", [])

            repo_instance = repo_mapper[repository_class]
            logger.info(f"Processing repository: {repository_class}")

            data = repo_instance.get_combined_table()
            data = data[features]

            logger.info(
                f"Fetched data for {repository_class}: {data.shape[0]} rows and {data.shape[1]} columns"
            )

            for strategy in strategies:
                strategy_name = strategy.get("name")
                params = strategy.get("params", {})
                model_save_path = strategy.get("model_save_path", None)

                logger.info(f"Applying strategy: {strategy_name} with params: {params}")
                model_trainer = ModelTrainer(
                    ModelTrainerConfig(
                        model_name=strategy_name,
                        params=params,
                        model_save_path=model_save_path,
                    )
                )
                logger.info(f"Training model for {strategy_name} on {repository_class}")

                # Before training (implemention remained)
                # 1. Find features to use for training (keep it in config once EDA is done)
                # - shape, describe, see distribution, see pairplot
                # 2. Handle NaNs, duplicates
                # 3. Convert categorical features to numerical if needed
                # 4. Normalize/scale features if needed
                # 5. Dimensionality reduction if needed
                # 6. Split data into training and testing sets if needed
                # 7. Train the model
                # 8. Save the model
                # 9. Visualize anomaly if possible
                model_trainer.train(data)
                logger.info(
                    f"Model trained and saved for {strategy_name} on {repository_class}"
                )

        logger.info("Stage-wise anomaly detection completed successfully.")


run_stage_wise_anomaly_detection()
