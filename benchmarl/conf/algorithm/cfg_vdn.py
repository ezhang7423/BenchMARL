

from dataclasses import dataclass
from typing import Type

from benchmarl.algorithms import Vdn
from benchmarl.algorithms.common import Algorithm
from benchmarl.conf.algorithm.cfg_common import AlgorithmConfig

@dataclass
class VdnConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Vdn`."""

    delay_value: bool = True
    loss_function: str = "l2"

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Vdn

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False

@dataclass
class nodelay(VdnConfig):
    delay_value: bool = False
