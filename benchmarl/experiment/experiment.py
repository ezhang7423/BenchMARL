#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

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
from benchmarl.conf.experiment.common import ExperimentConfig
from benchmarl.environments import Task
from benchmarl.experiment.callback import Callback, CallbackNotifier
from benchmarl.experiment.logger import Logger
from benchmarl.models.common import ModelConfig
from eztils.torch import seed_everything


class Experiment(CallbackNotifier):
    """
    Main experiment class in BenchMARL.

    Args:
        task (Task): the task configuration
        algorithm_config (AlgorithmConfig): the algorithm configuration
        model_config (ModelConfig): the policy model configuration
        seed (int): the seed for the experiment
        config (ExperimentConfig): the experiment config
        critic_model_config (ModelConfig, optional): the policy model configuration.
            If None, it defaults to model_config
        callbacks (list of Callback, optional): callbacks for this experiment
    """

    def __init__(
        self,
        task: Task,
        algorithm_config: AlgorithmConfig,
        model_config: ModelConfig,
        seed: int,
        config: ExperimentConfig,
        critic_model_config: Optional[ModelConfig] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        super().__init__(
            experiment=self, callbacks=callbacks if callbacks is not None else []
        )

        self.config = config

        self.task = task
        self.model_config = model_config
        self.critic_model_config = (
            critic_model_config if critic_model_config is not None else model_config
        )
        self.algorithm_config = algorithm_config
        self.seed = seed

        self._setup()

        self.total_time = 0
        self.total_frames = 0
        self.n_iters_performed = 0
        self.mean_return = 0

        if self.config.restore_file is not None:
            self._load_experiment()

    @property
    def on_policy(self) -> bool:
        """Whether the algorithm has to be run on policy."""
        return self.algorithm_config.on_policy()

    def _setup(self):
        self.config.validate(self.on_policy)
        seed_everything(self.seed)
        self._set_action_type()
        self._setup_task()
        self._setup_algorithm()
        self._setup_collector()
        self._setup_name()
        self._setup_logger()
        self._on_setup()

    def _set_action_type(self):
        if (
            self.task.supports_continuous_actions()
            and self.algorithm_config.supports_continuous_actions()
            and self.config.prefer_continuous_actions
        ):
            self.continuous_actions = True
        elif (
            self.task.supports_discrete_actions()
            and self.algorithm_config.supports_discrete_actions()
        ):
            self.continuous_actions = False
        elif (
            self.task.supports_continuous_actions()
            and self.algorithm_config.supports_continuous_actions()
        ):
            self.continuous_actions = True
        else:
            raise ValueError(
                f"Algorithm {self.algorithm_config} is not compatible"
                f" with the action space of task {self.task} "
            )

    def _setup_task(self):
        test_env = self.model_config.process_env_fun(
            self.task.get_env_fun(
                num_envs=self.config.evaluation_episodes,
                continuous_actions=self.continuous_actions,
                seed=self.seed,
                device=self.config.sampling_device,
            )
        )()
        env_func = self.model_config.process_env_fun(
            self.task.get_env_fun(
                num_envs=self.config.n_envs_per_worker(self.on_policy),
                continuous_actions=self.continuous_actions,
                seed=self.seed,
                device=self.config.sampling_device,
            )
        )

        self.observation_spec = self.task.observation_spec(test_env)
        self.info_spec = self.task.info_spec(test_env)
        self.state_spec = self.task.state_spec(test_env)
        self.action_mask_spec = self.task.action_mask_spec(test_env)
        self.action_spec = self.task.action_spec(test_env)
        self.group_map = self.task.group_map(test_env)
        self.train_group_map = copy.deepcopy(self.group_map)
        self.max_steps = self.task.max_steps(test_env)

        transforms = [self.task.get_reward_sum_transform(test_env)]
        transform = Compose(*transforms)

        if test_env.batch_size == ():
            self.env_func = lambda: TransformedEnv(
                SerialEnv(self.config.n_envs_per_worker(self.on_policy), env_func),
                transform.clone(),
            )
        else:
            self.env_func = lambda: TransformedEnv(env_func(), transform.clone())

        self.test_env = test_env.to(self.config.sampling_device)

    def _setup_algorithm(self):
        self.algorithm = self.algorithm_config.get_algorithm(experiment=self)
        self.replay_buffers = {
            group: self.algorithm.get_replay_buffer(
                group=group,
            )
            for group in self.group_map.keys()
        }
        self.losses = {
            group: self.algorithm.get_loss_and_updater(group)[0]
            for group in self.group_map.keys()
        }
        self.target_updaters = {
            group: self.algorithm.get_loss_and_updater(group)[1]
            for group in self.group_map.keys()
        }
        self.optimizers = {
            group: {
                loss_name: torch.optim.Adam(
                    params, lr=self.config.lr, eps=self.config.adam_eps
                )
                for loss_name, params in self.algorithm.get_parameters(group).items()
            }
            for group in self.group_map.keys()
        }

    def _setup_collector(self):
        self.policy = self.algorithm.get_policy_for_collection()

        self.group_policies = {}
        for group in self.group_map.keys():
            group_policy = self.policy.select_subsequence(out_keys=[(group, "action")])
            assert len(group_policy) == 1
            self.group_policies.update({group: group_policy[0]})

        self.collector = SyncDataCollector(
            self.env_func,
            self.policy,
            device=self.config.sampling_device,
            storing_device=self.config.train_device,
            frames_per_batch=self.config.collected_frames_per_batch(self.on_policy),
            total_frames=self.config.get_max_n_frames(self.on_policy),
            init_random_frames=(
                self.config.off_policy_init_random_frames if not self.on_policy else 0
            ),
        )

    def _setup_name(self):
        self.algorithm_name = self.algorithm_config.associated_class().__name__.lower()
        self.model_name = self.model_config.associated_class().__name__.lower()
        self.environment_name = self.task.env_name().lower()
        self.task_name = self.task.name.lower()

        if self.config.restore_file is not None and self.config.save_folder is not None:
            raise ValueError(
                "Experiment restore file and save folder have both been specified."
                "Do not set a save_folder when you are reloading an experiment as"
                "it will by default reloaded into the old folder."
            )
        if self.config.restore_file is None:
            if self.config.save_folder is not None:
                folder_name = Path(self.config.save_folder)
            else:
                folder_name = Path(os.getcwd())
            self.name = generate_exp_name(
                f"{self.algorithm_name}_{self.task_name}_{self.model_name}", ""
            )
            self.folder_name = folder_name / self.name
            if (
                len(self.config.loggers)
                or self.config.checkpoint_interval > 0
                or self.config.create_json
            ):
                self.folder_name.mkdir(parents=False, exist_ok=False)
        else:
            self.folder_name = Path(self.config.restore_file).parent.parent.resolve()
            self.name = self.folder_name.name

    def _setup_logger(self):
        self.logger = Logger(
            experiment_name=self.name,
            folder_name=str(self.folder_name),
            experiment_config=self.config,
            algorithm_name=self.algorithm_name,
            model_name=self.model_name,
            environment_name=self.environment_name,
            task_name=self.task_name,
            group_map=self.group_map,
            seed=self.seed,
        )
        self.logger.log_hparams(
            experiment_config=self.config.__dict__,
            algorithm_config=self.algorithm_config.__dict__,
            model_config=self.model_config.__dict__,
            task_config=self.task.config,
            continuous_actions=self.continuous_actions,
            on_policy=self.on_policy,
        )

    def run(self):
        """Run the experiment until completion."""
        try:
            torch.cuda.empty_cache()
            self._collection_loop()
        except KeyboardInterrupt as interrupt:
            print("\n\nExperiment was closed gracefully\n\n")
            self.close()
            raise interrupt
        except Exception as err:
            print("\n\nExperiment failed and is closing gracefully\n\n")
            self.close()
            raise err

    def _collection_loop(self):
        pbar = tqdm(
            initial=self.n_iters_performed,
            total=self.config.get_max_n_iters(self.on_policy),
        )
        sampling_start = time.time()

        # Training/collection iterations
        for batch in self.collector:  #!! important
            # Logging collection
            collection_time = time.time() - sampling_start
            current_frames = batch.numel()
            self.total_frames += current_frames
            self.mean_return = self.logger.log_collection(
                batch,
                total_frames=self.total_frames,
                task=self.task,
                step=self.n_iters_performed,
            )
            pbar.set_description(f"mean return = {self.mean_return}", refresh=False)

            # Callback
            self._on_batch_collected(batch)

            # Loop over groups
            training_start = time.time()
            for group in self.train_group_map.keys():
                group_batch = batch.exclude(*self._get_excluded_keys(group))
                group_batch = self.algorithm.process_batch(group, group_batch)
                group_batch = group_batch.reshape(-1)
                self.replay_buffers[group].extend(group_batch)

                training_tds = []
                for _ in range(self.config.n_optimizer_steps(self.on_policy)):
                    for _ in range(
                        self.config.train_batch_size(self.on_policy)
                        // self.config.train_minibatch_size(self.on_policy)
                    ):
                        training_tds.append(self._optimizer_loop(group))  #!! important
                training_td = torch.stack(training_tds)
                self.logger.log_training(
                    group, training_td, step=self.n_iters_performed
                )

                # Callback
                self._on_train_end(training_td, group)  #!!

                # Exploration update
                if isinstance(self.group_policies[group], TensorDictSequential):
                    explore_layer = self.group_policies[group][-1]
                else:
                    explore_layer = self.group_policies[group]
                if hasattr(explore_layer, "step"):  # Step exploration annealing
                    explore_layer.step(current_frames)  #!! important

            # Update policy in collector
            self.collector.update_policy_weights_()

            # Timers
            training_time = time.time() - training_start
            iteration_time = collection_time + training_time
            self.total_time += iteration_time
            self.logger.log(
                {
                    "timers/collection_time": collection_time,
                    "timers/training_time": training_time,
                    "timers/iteration_time": iteration_time,
                    "timers/total_time": self.total_time,
                    "counters/current_frames": current_frames,
                    "counters/total_frames": self.total_frames,
                    "counters/iter": self.n_iters_performed,
                },
                step=self.n_iters_performed,
            )

            # Evaluation
            if (
                self.config.evaluation
                and (self.total_frames % self.config.evaluation_interval == 0)
                and (len(self.config.loggers) or self.config.create_json)
            ):
                self._evaluation_loop()  #!! important

            # End of step
            self.n_iters_performed += 1
            self.logger.commit()
            if (
                self.config.checkpoint_interval > 0
                and self.total_frames % self.config.checkpoint_interval == 0
            ):
                self._save_experiment()  #!! important
            pbar.update()
            sampling_start = time.time()

        self.close()

    def close(self):
        """Close the experiment."""
        self.collector.shutdown()
        self.test_env.close()
        self.logger.finish()

    def _get_excluded_keys(self, group: str):
        excluded_keys = []
        for other_group in self.group_map.keys():
            if other_group != group:
                excluded_keys += [other_group, ("next", other_group)]
        excluded_keys += ["info", (group, "info"), ("next", group, "info")]
        return excluded_keys

    def _optimizer_loop(self, group: str) -> TensorDictBase:
        subdata = self.replay_buffers[group].sample()
        loss_vals = self.losses[group](subdata)
        training_td = loss_vals.detach()
        loss_vals = self.algorithm.process_loss_vals(group, loss_vals)

        for loss_name, loss_value in loss_vals.items():
            if loss_name in self.optimizers[group].keys():
                optimizer = self.optimizers[group][loss_name]

                loss_value.backward()

                grad_norm = self._grad_clip(optimizer)

                training_td.set(
                    f"grad_norm_{loss_name}",
                    torch.tensor(grad_norm, device=self.config.train_device),
                )

                optimizer.step()
                optimizer.zero_grad()
        self.replay_buffers[group].update_tensordict_priority(subdata)
        if self.target_updaters[group] is not None:
            self.target_updaters[group].step()

        callback_loss = self._on_train_step(subdata, group)
        if callback_loss is not None:
            training_td.update(callback_loss)

        return training_td

    def _grad_clip(self, optimizer: torch.optim.Optimizer) -> float:
        params = []
        for param_group in optimizer.param_groups:
            params += param_group["params"]

        if self.config.clip_grad_norm and self.config.clip_grad_val is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                params, self.config.clip_grad_val
            )
        else:
            norm_type = 2.0
            norms = [
                torch.linalg.vector_norm(p.grad, norm_type)
                for p in params
                if p.grad is not None
            ]
            total_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type)
            if self.config.clip_grad_val is not None:
                torch.nn.utils.clip_grad_value_(params, self.config.clip_grad_val)

        return float(total_norm)

    @torch.no_grad()
    def _evaluation_loop(self):
        evaluation_start = time.time()
        with set_exploration_type(
            ExplorationType.MODE
            if self.config.evaluation_deterministic_actions
            else ExplorationType.RANDOM
        ):
            if self.task.has_render(self.test_env) and self.config.render:
                video_frames = []

                def callback(env, td):
                    video_frames.append(
                        self.task.__class__.render_callback(self, env, td)
                    )

            else:
                video_frames = None
                callback = None

            if self.test_env.batch_size == ():
                rollouts = []
                for eval_episode in range(self.config.evaluation_episodes):
                    rollouts.append(
                        self.test_env.rollout(
                            max_steps=self.max_steps,
                            policy=self.policy,
                            callback=callback if eval_episode == 0 else None,
                            auto_cast_to_device=True,
                            break_when_any_done=True,
                        )
                    )
            else:
                rollouts = self.test_env.rollout(
                    max_steps=self.max_steps,
                    policy=self.policy,
                    callback=callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
                )
                rollouts = list(rollouts.unbind(0))
        evaluation_time = time.time() - evaluation_start
        self.logger.log(
            {"timers/evaluation_time": evaluation_time}, step=self.n_iters_performed
        )
        self.logger.log_evaluation(
            rollouts,
            video_frames=video_frames,
            step=self.n_iters_performed,
            total_frames=self.total_frames,
        )
        # Callback
        self._on_evaluation_end(rollouts)

    # Saving experiment state
    def state_dict(self) -> OrderedDict:
        """Get the state_dict for the experiment."""
        state = OrderedDict(
            total_time=self.total_time,
            total_frames=self.total_frames,
            n_iters_performed=self.n_iters_performed,
            mean_return=self.mean_return,
        )
        state_dict = OrderedDict(
            state=state,
            collector=self.collector.state_dict(),
            **{f"loss_{k}": item.state_dict() for k, item in self.losses.items()},
            **{
                f"buffer_{k}": item.state_dict()
                for k, item in self.replay_buffers.items()
            },
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load the state_dict for the experiment.

        Args:
            state_dict (dict): the state dict

        """
        for group in self.group_map.keys():
            self.losses[group].load_state_dict(state_dict[f"loss_{group}"])
            self.replay_buffers[group].load_state_dict(state_dict[f"buffer_{group}"])
        self.collector.load_state_dict(state_dict["collector"])
        self.total_time = state_dict["state"]["total_time"]
        self.total_frames = state_dict["state"]["total_frames"]
        self.n_iters_performed = state_dict["state"]["n_iters_performed"]
        self.mean_return = state_dict["state"]["mean_return"]

    def _save_experiment(self) -> None:
        """Checkpoint trainer"""
        checkpoint_folder = self.folder_name / "checkpoints"
        checkpoint_folder.mkdir(parents=False, exist_ok=True)
        checkpoint_file = checkpoint_folder / f"checkpoint_{self.total_frames}.pt"
        torch.save(self.state_dict(), checkpoint_file)

    def _load_experiment(self) -> Experiment:
        """Load trainer from checkpoint"""
        loaded_dict: OrderedDict = torch.load(self.config.restore_file)
        self.load_state_dict(loaded_dict)
        return self
