"""
Other global variables
"""

import dataclasses
import importlib
import json
import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

from dotenv import load_dotenv
from eztils import abspath, datestr, setup_path
from eztils.argparser import HfArgumentParser, update_dataclass_defaults
from rich import print

from benchmarl.lib.experiment.experiment import Experiment

load_dotenv()


def get_version() -> str:
    try:
        return importlib_metadata.version("benchmarl")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
__version__ = version

REPO_DIR = setup_path(Path(abspath()) / "..")
DATA_ROOT = setup_path(os.getenv("DATA_ROOT") or REPO_DIR)
RUN_DIR = LOG_DIR = Path()


@dataclass
class DefaultParams:  # the config for this file.
    experiment: str = "common.ExperimentConfig"
    algorithm: str = "cfg_vdn.VdnConfig"
    task: str = "vmas.balance.TaskConfig"
    model: str = "mlp.MlpConfig"
    critic_model: str = "mlp.MlpConfig"
    seed: int = 42
    wandb: bool = False
    # nested: Nest = field(default_factory=Nest) # TODO alternatively than current method, support nested dataclasses in the future (make a better hydra)


def import_module(prefix, path):
    *module_path, attr = path.split(".")
    return getattr(importlib.import_module(f"{prefix}.{'.'.join(module_path)}"), attr)


def setup_experiment():
    """
    Sets up the experiment by creating a run directory and a log directory, and creating a symlink from the repo directory to the run directory.
    """
    print("Setting up experiment...")
    global RUN_DIR
    global LOG_DIR

    # create run dir
    RUN_DIR = setup_path(DATA_ROOT / "runs")
    LOG_DIR = setup_path(RUN_DIR / datestr())

    print(f"LOG DIR: {LOG_DIR}")

    # symlink repo dir / runs to run_dir
    if not (REPO_DIR / "runs").exists() and (REPO_DIR / "runs") != RUN_DIR:
        print(f'Creating symlink from {REPO_DIR / "runs"} to {RUN_DIR}')
        (REPO_DIR / "runs").symlink_to(RUN_DIR)

    os.chdir(LOG_DIR)

    """
    SETUP CONFIG
    precedence order goes config_file < algorithm/task < specific parameters
    """
    # TODO generalize to support an arbitrary amount of default parameters rather than just algorithm and task (use for loops and dictionary)

    # setup algorithm and task defaults
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str
    )  # a folder with the respective config.jsons

    args, *_ = parser.parse_known_args()
    config_file = args.config
    if config_file is not None:
        with open(config_file, "r") as config:
            config = json.load(config)
        DefaultParams.algorithm = config["_algorithm"]
        DefaultParams.task = config["_task"]

    # get algorithma and task configs
    parser.add_argument("-a", "--algorithm", type=str, default=DefaultParams.algorithm)
    parser.add_argument("-t", "--task", type=str, default=DefaultParams.task)
    args, *_ = parser.parse_known_args()

    algorithm_config = import_module("benchmarl.conf.algorithm", args.algorithm)
    task_config = import_module("benchmarl.conf.environment", args.task)

    # now parse the actual dataclasses specified by the configs
    parser = HfArgumentParser((algorithm_config, task_config))
    parser.print_help(sys.stdout)

    *conf, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # python benchmarl -a cfg_vdn/NoDelayVdnConfig -t vmas/balance

    if (
        config_file is not None
    ):  # overwrite the default alg and task params according to the config file
        og_algorithm_config, og_task_config = parser.parse_json_file(
            config_file, allow_extra_keys=True
        )
        if (
            DefaultParams.algorithm == config["_algorithm"]
        ):  # only overwrite default alg params with config if user didn't specify new algorithm
            algorithm_config = update_dataclass_defaults(
                algorithm_config, og_algorithm_config
            )
        if (
            DefaultParams.task == config["_task"]
        ):  # only overwrite default task params with config if user didn't specify new task
            task_config = update_dataclass_defaults(task_config, og_task_config)

        parser = HfArgumentParser((algorithm_config, task_config))
        *conf, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    parser.to_json(conf, LOG_DIR / "config.json")

    # add the algorithm and task
    with open(LOG_DIR / "config.json", "r") as config:
        serialized_cfg = json.load(config)
        serialized_cfg["_algorithm"] = args.algorithm
        serialized_cfg["_task"] = args.task
    with open(LOG_DIR / "config.json", "w") as config:
        json.dump(serialized_cfg, config)

    return conf


def main():
    algorithm_config, task_config = conf = setup_experiment()
    from benchmarl.conf.environment import task_config_registry

    task_key = ".".join(DefaultParams.task.split(".")[:-1])
    experiment = Experiment(
        task=task_config_registry[task_key].update_config(
            dataclasses.asdict(task_config)
        ),
        algorithm_config=algorithm_config,
        seed=DefaultParams.seed,  # TODO put this as part of experiment config
        model_config=import_module("benchmarl.conf.model", DefaultParams.model)(),
        critic_model_config=import_module(
            "benchmarl.conf.model", DefaultParams.critic_model
        )(),
        config=import_module("benchmarl.conf.experiment", DefaultParams.experiment)(),
    )
    print(f"[bold green]Welcome to benchmarl v{version}[/]")
    print(conf)

    experiment.run()


if __name__ == "__main__":
    main()
