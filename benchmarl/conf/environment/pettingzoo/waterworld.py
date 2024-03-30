#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    task: str = "waterworld_v4"
    max_cycles: int = 500
    n_pursuers: int = 2
    n_evaders: int = 5
    n_poisons: int = 10
    n_obstacles: int = 1
    n_coop: int = 1
    n_sensors: int = 30
    sensor_range: float = 0.2
    radius: float = 0.015
    obstacle_radius: float = 0.1
    pursuer_max_accel: float = 0.5
    pursuer_speed: float = 0.2
    evader_speed: float = 0.1
    poison_speed: float = 0.1
    poison_reward: float = -1.0
    food_reward: float = 10.0
    encounter_reward: float = 0.01
    thrust_penalty: float = -0.5
    local_ratio: float = 1.0
    speed_features: bool = True
