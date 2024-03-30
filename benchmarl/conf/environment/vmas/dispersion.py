#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    max_steps: int = 100
    n_agents: int = 4
    n_food: int = 4
    share_rew: bool = True
    food_radius: float = 0.02
    penalise_by_time: bool = False
