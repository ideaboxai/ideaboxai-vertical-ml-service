from .strategies.base import BaseScoringRankingStrategy
from .strategies.topsis_ranker import TOPSISRankerStrategy


def get_ranker(strategy_name: str, params: dict = None) -> BaseScoringRankingStrategy:
    rankers = {
        "topsis": TOPSISRankerStrategy,
    }
    ranker_class = rankers.get(strategy_name)
    if not ranker_class:
        raise ValueError(f"Ranker '{strategy_name}' not recognized.")
    return ranker_class(params=params)
