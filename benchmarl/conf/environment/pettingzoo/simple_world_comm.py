#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    task: str = "simple_world_comm_v3"
    max_cycles: int = 100
    num_good: int = 2
    num_adversaries: int = 4
    num_obstacles: int = 1
    num_food: int = 2
    num_forests: int = 2
