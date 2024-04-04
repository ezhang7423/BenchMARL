#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from typing import Callable, Dict, List, Optional

from benchmarl.conf.environment import Task
from benchmarl.lib.utils import DEVICE_TYPING

from torchrl.data import CompositeSpec
from torchrl.envs.common import EnvBase
from torchrl.envs.libs.meltingpot import MeltingpotEnv


class MeltingPotTask(Task):
    """Enum for MeltingPot tasks."""

    HARVEST_OPEN = None # these enums all get mapped to the config dict. e.g. {'task': 'commons_harvest__open', 'max_cycles': 200}
    
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        if self.supports_continuous_actions() and self.supports_discrete_actions():
            self.config.update({"continuous_actions": continuous_actions})


        # substrate: str | "ml_collections.config_dict.ConfigDict",  # noqa        
        # max_steps: Optional[int] = None,
        # categorical_actions: bool = True,
        # group_map: MarlGroupMapType
        
        return lambda: MeltingpotEnv(
            substrate=self.config['task'],
            categorical_actions=True,
            device=device,
            # seed=seed, TODO support this
            # render_mode="rgb_array",
            # **self.config
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_state(self) -> bool:        
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_cycles"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        if "state" in env.observation_spec:
            return CompositeSpec({"state": env.observation_spec["state"].clone()})
        return None
    
    
    #############3
    # TODO need to test
    #############3
    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "action_mask":
                    del group_obs_spec[key]
            if group_obs_spec.is_empty():
                del observation_spec[group]
        if "state" in observation_spec.keys():
            del observation_spec["state"]
        if observation_spec.is_empty():
            return None

        return observation_spec

    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
                observation_spec[group]['observation'] = observation_spec[group]['observation', 'RGB']

        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        return observation_spec

    def action_spec(self, env: EnvBase) -> CompositeSpec:
        return env.input_spec["full_action_spec"]

    @staticmethod
    def env_name() -> str:
        return "meltingpot"
