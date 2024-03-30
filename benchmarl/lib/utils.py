#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import importlib
from dataclasses import field
from typing import Any, Dict, Union

import torch
import yaml


def _class_from_name(name: str):
    name_split = name.split(".")
    module_name = ".".join(name_split[:-1])
    class_name = name_split[-1]
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def _read_yaml_config(config_file: str) -> Dict[str, Any]:  # TODO get rid of all yaml
    with open(config_file) as config:
        yaml_string = config.read()
    config_dict = yaml.safe_load(yaml_string)
    if "defaults" in config_dict.keys():
        del config_dict["defaults"]
    return config_dict


def df(fun):
    return field(default_factory=fun)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


DEVICE_TYPING = Union[torch.device, str, int]
