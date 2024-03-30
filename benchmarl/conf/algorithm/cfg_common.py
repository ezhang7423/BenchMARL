from abc import abstractmethod
from dataclasses import dataclass
from typing import Type

from benchmarl.lib.algorithms.common import Algorithm


@dataclass
class AlgorithmConfig:
    """
    Dataclass representing an algorithm configuration.
    This should be overridden by implemented algorithms.
    Implementors should:

        1. add configuration parameters for their algorithm
        2. implement all abstract methods

    """

    def get_algorithm(self, experiment) -> Algorithm:
        """
        Main function to turn the config into the associated algorithm

        Args:
            experiment (Experiment): the experiment class

        Returns: the Algorithm

        """
        return self.associated_class()(
            **self.__dict__,  # Passes all the custom config parameters
            experiment=experiment,
        )

    @staticmethod
    @abstractmethod
    def associated_class() -> Type[Algorithm]:
        """
        The algorithm class associated to the config
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def on_policy() -> bool:
        """
        If the algorithm has to be run on policy or off policy
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supports_continuous_actions() -> bool:
        """
        If the algorithm supports continuous actions
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supports_discrete_actions() -> bool:
        """
        If the algorithm supports discrete actions
        """
        raise NotImplementedError
