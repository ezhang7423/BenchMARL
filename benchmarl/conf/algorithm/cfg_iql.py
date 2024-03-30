from dataclasses import dataclass
from typing import Type

from benchmarl.conf.algorithm.cfg_common import AlgorithmConfig

from benchmarl.lib.algorithms import Iql
from benchmarl.lib.algorithms.common import Algorithm


@dataclass
class IqlConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Iql`."""

    delay_value: bool = True
    loss_function: str = "l2"

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Iql

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False
