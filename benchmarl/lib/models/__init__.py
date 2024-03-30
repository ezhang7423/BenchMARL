#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from benchmarl.conf.model.gnn import GnnConfig
from benchmarl.conf.model.mlp import MlpConfig

from .common import Model, ModelConfig, SequenceModel, SequenceModelConfig
from .gnn import Gnn
from .mlp import Mlp

classes = ["Mlp", "MlpConfig", "Gnn", "GnnConfig"]

model_config_registry = {"mlp": MlpConfig, "gnn": GnnConfig}

__all__ = [Model, ModelConfig, SequenceModel, SequenceModelConfig, Gnn, Mlp]
