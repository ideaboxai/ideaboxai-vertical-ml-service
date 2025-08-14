from abc import ABC, abstractmethod


class BaseRepository(ABC):
    """
    Abstract base class for repositories that handle data preparation.
    """

    @abstractmethod
    def get_individual_table(self):
        pass

    @abstractmethod
    def get_combined_table(self):
        pass
