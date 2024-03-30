
from __future__ import annotations

import copy
import importlib

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, MISSING
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.envs import SerialEnv, TransformedEnv
from torchrl.envs.transforms import Compose
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name
from tqdm import tqdm

from benchmarl.conf.algorithm.cfg_common import AlgorithmConfig
from benchmarl.environments import Task
from benchmarl.experiment.callback import Callback, CallbackNotifier
from benchmarl.experiment.logger import Logger
from benchmarl.models.common import ModelConfig
from benchmarl.utils import df, list_field

@dataclass
class ExperimentConfig:
    """
    Configuration class for experiments.
    This class acts as a schema for loading and validating yaml configurations.

    Parameters in this class aim to be agnostic of the algorithm, task or model used.
    To know their meaning, please check out the descriptions in ``benchmarl/conf/experiment/base_experiment.yaml``
    """


    # The device for collection (e.g. cuda)
    sampling_device: str = "cuda"
    # The device for training (e.g. cuda)
    train_device: str = "cuda"

    # Whether to share the parameters of the policy within agent groups
    share_policy_params: bool = True
    # If an algorithm and an env support both continuous and discrete actions, what should be preferred
    prefer_continuous_actions: bool = True

    # Discount factor
    gamma: float = 0.9
    # Learning rate
    lr: float = 0.00005
    # The epsilon parameter of the adam optimizer
    adam_eps: float = 0.000001
    # Clips grad norm if true and clips grad value if false
    clip_grad_norm: bool = True
    # The value for the clipping, if null no clipping
    clip_grad_val: Optional[float] = 5

    # Whether to use soft or hard target updates
    soft_target_update: bool = True
    # If soft_target_update is True, this is its polyak_tau
    polyak_tau: float = 0.005
    # If soft_target_update is False, this is the frequency of the hard trarget updates in terms of n_optimizer_steps
    hard_target_update_frequency: int = 5

    # When an exploration wrapper is used. This is its initial epsilon for annealing
    exploration_eps_init: float = 0.8
    # When an exploration wrapper is used. This is its final epsilon after annealing
    exploration_eps_end: float = 0.01
    # Number of frames for annealing of exploration strategy in deterministic policy algorithms
    # If null it will default to max_n_frames / 3
    exploration_anneal_frames: Optional[int] = None

    # The maximum number of experiment iterations before the experiment terminates, exclusive with max_n_frames
    max_n_iters: Optional[int] = None
    # Number of collected frames before ending, exclusive with max_n_iters
    max_n_frames: Optional[int] = 3_000_000

    # Number of frames collected and each experiment iteration
    on_policy_collected_frames_per_batch: int = 6000
    # Number of environments used for collection
    # If the environment is vectorized, this will be the number of batched environments.
    # Otherwise batching will be simulated and each env will be run sequentially.
    on_policy_n_envs_per_worker: int = 10
    # This is the number of times collected_frames_per_batch will be split into minibatches and trained
    on_policy_n_minibatch_iters: int = 45
    # In on-policy algorithms the train_batch_size will be equal to the on_policy_collected_frames_per_batch
    # and it will be split into minibatches with this number of frames for training
    on_policy_minibatch_size: int = 400

    # Number of frames collected and each experiment iteration
    off_policy_collected_frames_per_batch: int = 6000
    # Number of environments used for collection
    # If the environment is vectorized, this will be the number of batched environments.
    # Otherwise batching will be simulated and each env will be run sequentially.
    off_policy_n_envs_per_worker: int = 10
    # This is the number of times off_policy_train_batch_size will be sampled from the buffer and trained over.
    off_policy_n_optimizer_steps: int = 1000
    # Number of frames used for each off_policy_n_optimizer_steps when training off-policy algorithms
    off_policy_train_batch_size: int = 128
    # Maximum number of frames to keep in replay buffer memory for off-policy algorithms
    off_policy_memory_size: int = 1_000_000
    # Number of random action frames to prefill the replay buffer with
    off_policy_init_random_frames: int = 0

    evaluation: bool = True
    # Whether to render the evaluation (if rendering is available)
    render: bool = True
    # Frequency of evaluation in terms of collected frames (this should be a multiple of on/off_policy_collected_frames_per_batch)
    evaluation_interval: int = 120_000
    # Number of episodes that evaluation is run on
    evaluation_episodes: int = 10
    # If True, when stochastic policies are evaluated, their mode is taken, otherwise, if False, they are sampled
    evaluation_deterministic_actions: bool = True

    # List of loggers to use, options are: wandb, csv, tensorboard, mflow
    loggers: List[str] = list_field(["csv"])
    # Create a json folder as part of the output in the format of marl-eval
    create_json: bool = True

    # Absolute path to the folder where the experiment will log.
    # If null, this will default to the hydra output dir (if using hydra) or to the current folder when the script is run (if not).
    save_folder: Optional[str] = None
    # Absolute path to a checkpoint file where the experiment was saved. If null the experiment is started fresh.
    restore_file: Optional[str] = None
    # Interval for experiment saving in terms of collected frames (this should be a multiple of on/off_policy_collected_frames_per_batch).
    # Set it to 0 to disable checkpointing
    checkpoint_interval: float = 300_000

    def train_batch_size(self, on_policy: bool) -> int:
        """
        The batch size of tensors used for training

        Args:
            on_policy (bool): is the algorithms on_policy

        """
        return (
            self.collected_frames_per_batch(on_policy)
            if on_policy
            else self.off_policy_train_batch_size
        )

    def train_minibatch_size(self, on_policy: bool) -> int:
        """
        The minibatch size of tensors used for training.
        On-policy algorithms are trained by splitting the train_batch_size (equal to the collected frames) into minibatches.
        Off-policy algorithms do not go through this process and thus have the ``train_minibatch_size==train_batch_size``

        Args:
            on_policy (bool): is the algorithms on_policy
        """
        return (
            self.on_policy_minibatch_size
            if on_policy
            else self.train_batch_size(on_policy)
        )

    def n_optimizer_steps(self, on_policy: bool) -> int:
        """
        Number of times to loop over the training step per collection iteration.

        Args:
            on_policy (bool): is the algorithms on_policy

        """
        return (
            self.on_policy_n_minibatch_iters
            if on_policy
            else self.off_policy_n_optimizer_steps
        )

    def replay_buffer_memory_size(self, on_policy: bool) -> int:
        """
        Size of the replay buffer memory in terms of frames

        Args:
            on_policy (bool): is the algorithms on_policy

        """
        return (
            self.collected_frames_per_batch(on_policy)
            if on_policy
            else self.off_policy_memory_size
        )

    def collected_frames_per_batch(self, on_policy: bool) -> int:
        """
        Number of collected frames per collection iteration.

         Args:
             on_policy (bool): is the algorithms on_policy

        """
        return (
            self.on_policy_collected_frames_per_batch
            if on_policy
            else self.off_policy_collected_frames_per_batch
        )

    def n_envs_per_worker(self, on_policy: bool) -> int:
        """
        Number of environments used for collection

        - In vectorized environments, this will be the vectorized batch_size.
        - In other environments, this will be emulated by running them sequentially.

        Args:
            on_policy (bool): is the algorithms on_policy


        """
        return (
            self.on_policy_n_envs_per_worker
            if on_policy
            else self.off_policy_n_envs_per_worker
        )

    def get_max_n_frames(self, on_policy: bool) -> int:
        """
        Get the maximum number of frames collected before the experiment ends.

        Args:
            on_policy (bool): is the algorithms on_policy
        """
        if self.max_n_frames is not None and self.max_n_iters is not None:
            return min(
                self.max_n_frames,
                self.max_n_iters * self.collected_frames_per_batch(on_policy),
            )
        elif self.max_n_frames is not None:
            return self.max_n_frames
        elif self.max_n_iters is not None:
            return self.max_n_iters * self.collected_frames_per_batch(on_policy)

    def get_max_n_iters(self, on_policy: bool) -> int:
        """
        Get the maximum number of experiment iterations before the experiment ends.

        Args:
            on_policy (bool): is the algorithms on_policy
        """
        return -(
            -self.get_max_n_frames(on_policy)
            // self.collected_frames_per_batch(on_policy)
        )

    def get_exploration_anneal_frames(self, on_policy: bool):
        """
        Get the number of frames for exploration annealing.
        If self.exploration_anneal_frames is None this will be a third of the total frames to collect.

        Args:
            on_policy (bool): is the algorithms on_policy
        """
        return (
            (self.get_max_n_frames(on_policy) // 3)
            if self.exploration_anneal_frames is None
            else self.exploration_anneal_frames
        )

    # @staticmethod
    # def get_from_yaml(path: Optional[str] = None):
    #     """
    #     Load the experiment configuration from yaml

    #     Args:
    #         path (str, optional): The full path of the yaml file to load from.
    #             If None, it will default to
    #             ``benchmarl/conf/experiment/base_experiment.yaml``

    #     Returns:
    #         the loaded :class:`~benchmarl.experiment.ExperimentConfig`
    #     """
    #     if path is None:
    #         yaml_path = (
    #             Path(__file__).parent.parent
    #             / "conf"
    #             / "experiment"
    #             / "base_experiment.yaml"
    #         )
    #         return ExperimentConfig(**_read_yaml_config(str(yaml_path.resolve())))
    #     else:
    #         return ExperimentConfig(**_read_yaml_config(path))

    def validate(self, on_policy: bool):
        """
        Validates config.

        Args:
            on_policy (bool): is the algorithms on_policy

        """
        if (
            self.evaluation
            and self.evaluation_interval % self.collected_frames_per_batch(on_policy)
            != 0
        ):
            raise ValueError(
                f"evaluation_interval ({self.evaluation_interval}) "
                f"is not a multiple of the collected_frames_per_batch ({self.collected_frames_per_batch(on_policy)})"
            )
        if (
            self.checkpoint_interval != 0
            and self.checkpoint_interval % self.collected_frames_per_batch(on_policy)
            != 0
        ):
            raise ValueError(
                f"checkpoint_interval ({self.checkpoint_interval}) "
                f"is not a multiple of the collected_frames_per_batch ({self.collected_frames_per_batch(on_policy)})"
            )
        if self.max_n_frames is None and self.max_n_iters is None:
            raise ValueError("n_iters and total_frames are both not set")

