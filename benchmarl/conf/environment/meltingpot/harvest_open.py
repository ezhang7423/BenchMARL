#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    task: str = "commons_harvest__open"
    max_cycles: int = 200 # max timesteps limit
    num_frames: int = 4
    
    