#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class TaskConfig:
    max_steps: int = 100
    n_agents: int = 4
    n_packages: int = 1
    package_width: float = 0.15
    package_length: float = 0.15
    package_mass: float = 50
