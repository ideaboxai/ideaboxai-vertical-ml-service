from abc import ABC, abstractmethod


class BaseScoringRankingStrategy(ABC):
    """Abstract base class for all scoring and ranking strategies."""

    def __init__(self, params: dict = None):
        self.params = params if params is not None else {}

    @abstractmethod
    def score_and_rank(self, dataset):
        """Method to be implemented by all concrete strategies for scoring and ranking."""
        pass
