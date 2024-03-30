
from dataclasses import dataclass
from typing import Type, Optional, Union

from benchmarl.algorithms import Masac
from benchmarl.algorithms.common import Algorithm
from benchmarl.conf.algorithm.cfg_common import AlgorithmConfig

@dataclass
class MasacConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Masac`."""

    share_param_critic: bool =  True
    num_qvalue_nets: int =  2
    loss_function: str =  "l2"
    delay_qvalue: bool =  True
    target_entropy: Union[float, str] =  "auto"
    discrete_target_entropy_weight: float =  0.2
    alpha_init: float =  1.0
    min_alpha: Optional[float] =  None
    max_alpha: Optional[float] =  None
    fixed_alpha: bool =  False
    scale_mapping: str =  "biased_softplus_1.0"
    use_tanh_normal: bool =  True

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Masac

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False
