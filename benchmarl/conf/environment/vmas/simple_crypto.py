#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    max_steps: int = 100
    dim_c: int = 4
