import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Type

from benchmarl.algorithms.common import Algorithm
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import (
    DiscreteTensorSpec,
    LazyTensorStorage,
    OneHotDiscreteTensorSpec,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import RandomSampler, SamplerWithoutReplacement
from torchrl.objectives import LossModule
from torchrl.objectives.utils import HardUpdate, SoftUpdate, TargetNetUpdater

from benchmarl.models.common import ModelConfig
from benchmarl.utils import _read_yaml_config, DEVICE_TYPING

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
    def _load_from_yaml(name: str) -> Dict[str, Any]:
        yaml_path = (
            pathlib.Path(__file__).parent.parent
            / "conf"
            / "algorithm"
            / f"{name.lower()}.yaml"
        )
        return _read_yaml_config(str(yaml_path.resolve()))

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        """
        Load the algorithm configuration from yaml

        Args:
            path (str, optional): The full path of the yaml file to load from.
                If None, it will default to
                ``benchmarl/conf/algorithm/self.associated_class().__name__``

        Returns: the loaded AlgorithmConfig
        """
        if path is None:
            return cls(
                **AlgorithmConfig._load_from_yaml(
                    name=cls.associated_class().__name__,
                )
            )
        else:
            return cls(**_read_yaml_config(path))

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
