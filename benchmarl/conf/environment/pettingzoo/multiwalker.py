#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    task: str = "multiwalker_v9"
    n_walkers: int = 3
    shared_reward: bool = False
    max_cycles: int = 500
    position_noise: float = 0.001
    angle_noise: float = 0.001
    forward_reward: float = 1.0
    fall_reward: float = -10
    terminate_reward: float = -100
    terminate_on_fall: bool = True
    remove_on_fall: bool = True
    terrain_length: int = 200
