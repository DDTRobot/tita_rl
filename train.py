import numpy as np
import os
from datetime import datetime
from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
from configs.tita_flat_config import TitaFlatCfg, TitaFlatCfgPPO
from configs.tita_rough_config import TitaRoughCfg, TitaRoughCfgPPO
from envs.no_constrains_legged_robot import Tita
from envs.single_stand import SingleStand
from configs.tita_single_stand_config import TitaSingleStandCfg,TitaSingleStandCfgPPO
from global_config import ROOT_DIR, ENVS_DIR
import isaacgym
from utils.helpers import get_args
from envs import LeggedRobot
from utils.task_registry import task_registry

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # Define the log path and task configuration folder path
    logs_path = os.path.join(ROOT_DIR, "logs")
    task_config_folder = os.path.join(logs_path, f"{args.task}")

    # Check if the task configuration folder exists and save configurations
    if os.path.exists(task_config_folder) and os.path.isdir(task_config_folder):
        print(f"Task configuration folder exists: {task_config_folder}, saving configuration files.")
        task_registry.save_cfgs(name=args.task, train_cfg=train_cfg)
    else:
        print(f"Task configuration folder does not exist: {task_config_folder}, skipping configuration saving.")

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':

    task_registry.register("tita_constraint",LeggedRobot,TitaConstraintRoughCfg(),TitaConstraintRoughCfgPPO())
    task_registry.register("tita_flat", Tita, TitaFlatCfg(), TitaFlatCfgPPO())
    task_registry.register("tita_rough", Tita, TitaRoughCfg(), TitaRoughCfgPPO())
    task_registry.register("single_stand", SingleStand, TitaSingleStandCfg(), TitaSingleStandCfgPPO())
    args = get_args()
    args.headless = True
    args.task = "single_stand"
    train(args)
