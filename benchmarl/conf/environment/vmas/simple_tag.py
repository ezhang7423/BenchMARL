#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    max_steps: int = 100
    num_good_agents: int = 1
    num_adversaries: int = 3
    num_landmarks: int = 2
    shape_agent_rew: bool = False
    shape_adversary_rew: bool = False
    agents_share_rew: bool = False
    adversaries_share_rew: bool = True
    observe_same_team: bool = True
    observe_pos: bool = True
    observe_vel: bool = True
    bound: float = 1.0
    respawn_at_catch: bool = False
