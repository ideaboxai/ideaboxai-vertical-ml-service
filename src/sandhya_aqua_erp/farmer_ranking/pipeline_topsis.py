import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from ml_services.scoring_ranking.compute_rank import (
    ScoringRanking,
    ScoringRankingConfig,
)

from src.sandhya_aqua_erp.repositories.grn_repo import GRNRepository

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_topsis_for_farmer_ranking(
    interval: str = "1 MONTH", exact_date_time: str = None
):
    with open("src/sandhya_aqua_erp/farmer_ranking/topsis_config.json", "r") as file:
        config_data = json.load(file)

    grn_repo = GRNRepository()
    farmer_data = grn_repo.get_farmer_related_data(
        interval=interval, exact_date_time=exact_date_time
    )

    ranking_config = ScoringRankingConfig(
        strategy_name=config_data["strategy_name"], params=config_data.get("params", {})
    )
    ranking_model = ScoringRanking(config=ranking_config)

    features = config_data["features"]
    feature_names = [f["name"] for f in features]
    impacts = [f["impact"] for f in features]
    weights = [f["weight"] for f in features]

    feature_data = farmer_data[["farmer"] + feature_names].copy()

    feature_data = feature_data.fillna(feature_data.mean())

    ranked_data = ranking_model.rank(
        data=feature_data, nCol=feature_data.shape[1], impact=impacts, weights=weights
    )

    ranked_data = farmer_data.merge(
        ranked_data[["farmer", "relative_closeness", "rank"]], on="farmer", how="left"
    )

    ranked_data = ranked_data.sort_values(by="rank")

    # output_path = config_data.get("output_path", "farmer_ranking_results.csv")
    # ranked_data.to_csv(output_path, index=False)
    # --------implementation to persist in the db
    # logger.info(f"Ranked data saved to {output_path}")
