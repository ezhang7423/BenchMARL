from dataclasses import dataclass
from typing import Type

from benchmarl.conf.algorithm.cfg_common import AlgorithmConfig

from benchmarl.lib.algorithms import Qmix
from benchmarl.lib.algorithms.common import Algorithm


@dataclass
class QmixConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Qmix`."""

    mixing_embed_dim: int = 32
    delay_value: bool = True
    loss_function: str = "l2"

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Qmix

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False
