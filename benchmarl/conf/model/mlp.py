from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Type

import torch

from benchmarl.lib.models.common import ModelConfig
from benchmarl.lib.models.mlp import Mlp
from benchmarl.lib.utils import list_field
from torch import nn


@dataclass
class MlpConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Mlp`."""

    num_cells: Sequence[int] = list_field([256, 256])
    layer_class: Type[nn.Module] = torch.nn.Linear

    activation_class: Type[nn.Module] = torch.nn.Tanh
    activation_kwargs: Optional[dict] = None

    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return Mlp
