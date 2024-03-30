from dataclasses import dataclass
from typing import Type

from benchmarl.conf.algorithm.cfg_common import AlgorithmConfig

from benchmarl.lib.algorithms import Iddpg
from benchmarl.lib.algorithms.common import Algorithm


@dataclass
class IddpgConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Iddpg`."""

    share_param_critic: bool = True
    loss_function: str = "l2"
    delay_value: bool = True
    use_tanh_mapping: bool = True

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Iddpg

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return False

    @staticmethod
    def on_policy() -> bool:
        return False
