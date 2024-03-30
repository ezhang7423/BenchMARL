from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

import torch_geometric

from benchmarl.lib.models.common import ModelConfig
from benchmarl.lib.models.gnn import Gnn
from benchmarl.lib.utils import df


@dataclass
class GnnConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Gnn`."""

    topology: str = "full"
    self_loops: bool = False

    gnn_class: Type[
        torch_geometric.nn.MessagePassing
    ] = torch_geometric.nn.conv.GraphConv
    gnn_kwargs: Optional[dict] = df(lambda: {"aggr": "add"})

    @staticmethod
    def associated_class():
        return Gnn
