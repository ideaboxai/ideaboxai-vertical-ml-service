import pandas as pd

from ml_services.scoring_ranking.factory import get_ranker

from dataclasses import dataclass

from typing import Optional


@dataclass
class ScoringRankingConfig:
    strategy_name: str
    params: Optional[dict] = None


class ScoringRanking:
    def __init__(self, config: ScoringRankingConfig):
        self.config = config
        self.strategy = get_ranker(
            strategy_name=self.config.strategy_name, params=self.config.params
        )

    def rank(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ranked_data = self.strategy.score_and_rank(data, **kwargs)
        return ranked_data
