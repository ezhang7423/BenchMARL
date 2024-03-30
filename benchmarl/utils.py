#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import importlib
import random
from typing import Any, Dict, Union

import torch
import yaml

_has_numpy = importlib.util.find_spec("numpy") is not None


DEVICE_TYPING = Union[torch.device, str, int]
