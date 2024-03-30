#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    task: str = "simple_spread_v3"
    max_cycles: int = 100
    local_ratio: float = 0.5
    N: int = 3
