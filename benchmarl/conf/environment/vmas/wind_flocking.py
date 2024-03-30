#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    max_steps: int = 100
    dist_shaping_factor: float = 1
    rot_shaping_factor: float = 0
    vel_shaping_factor: float = 1
    pos_shaping_factor: float = 0
    energy_shaping_factor: float = 0
    wind_shaping_factor: float = 1
    wind: float = 0
    cover_angle_tolerance: float = 1
    horizon: int = 100
    observe_rel_pos: bool = True
    observe_rel_vel: bool = True
    observe_pos: bool = False
    desired_vel: float = 0.4
