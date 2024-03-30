
from dataclasses import dataclass
from typing import Type

from benchmarl.algorithms import Ippo
from benchmarl.algorithms.common import Algorithm
from benchmarl.conf.algorithm.cfg_common import AlgorithmConfig


@dataclass
class IppoConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Ippo`."""

    share_param_critic: bool =  True
    clip_epsilon: float =  0.2
    entropy_coef: float =  0.0
    critic_coef: float =  1.0
    loss_critic_type: str =  "l2"
    lmbda: float =  0.9
    scale_mapping: str =  "biased_softplus_1.0"
    use_tanh_normal: bool =  True

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Ippo

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True
