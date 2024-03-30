#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    max_steps: int = 100
    n_agents: int = 3
    shared_rew: bool = False
    n_gaussians: int = 3
    lidar_range: float = 0.2
    cov: float = 0.05
    collisions: bool = True
    spawn_same_pos: bool = False
