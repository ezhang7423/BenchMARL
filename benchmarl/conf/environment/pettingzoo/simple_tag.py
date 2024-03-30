#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    task: str = "simple_tag_v3"
    num_good: int = 2
    num_adversaries: int = 3
    num_obstacles: int = 2
    max_cycles: int = 100
