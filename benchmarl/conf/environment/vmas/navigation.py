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
    collisions: bool = True
    agents_with_same_goal: int = 1
    observe_all_goals: bool = False
    shared_rew: bool = False
    split_goals: bool = False
    lidar_range: float = 0.35
    agent_radius: float = 0.1
