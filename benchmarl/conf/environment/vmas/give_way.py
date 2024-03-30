#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    max_steps: int = 100
    mirror_passage: bool = False
    observe_rel_pos: bool = False
    done_on_completion: bool = False
    final_reward: float = 0.01
