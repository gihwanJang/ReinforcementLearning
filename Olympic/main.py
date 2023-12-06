import os
import sys
import numpy as np
import torch
import random
import argparse
import gymnasium as gym

from pathlib import Path
project_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_path)

from ppo import PPOAgent
from wrapper import CompetitionOlympicsRunningEnvWrapper
from wrapper import CompetitionOlympicsWrestlingEnvWrapper
from config import running_args
from config import wrestling_args
from termproject_olympic.env.chooseenv import make

DEVICE = None

class GlobalConfig:
    def __init__(self):
        self.seed = 555
        self.path2save_train_history = "train_history"

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

config = GlobalConfig()
seed_everything(config.seed)


def make_env(env_name, agent=None, config=None):
    # environment
    env = make(env_type="olympics-integrated", game_name=env_name)
    if env_name == "olympics-running":
        env = CompetitionOlympicsRunningEnvWrapper(env, args=config)
    elif env_name == "olympics-wrestling":
        env = CompetitionOlympicsWrestlingEnvWrapper(env, args=config)

    return env

def main(args, evaluation=False):
    if not evaluation:
        # ppo agent
        agent = PPOAgent(make_env, args)
        # ppo train
        agent.train()
    else:
        agent = PPOAgent(make_env, args)
        agent.load_predtrain_model(f"{args.path2save_train_history}/actor.pth", f"{args.path2save_train_history}/critic.pth")
        agent.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", default="olympics-running")
    # parser.add_argument("--env_type", default="olympics-wrestling")
    # "olympics-wrestling", "olympics-running"
    args, rest_args = parser.parse_known_args()
    env_name = args.env_type

    if env_name == "olympics-running":
        args = running_args.get_args(rest_args)
    elif env_name == "olympics-wrestling":
        args = wrestling_args.get_args(rest_args)
    # elif env_name == "olympics-integrated":
    #     args = running_args.get_args(rest_args)
    else:
        raise Exception("Invalid Environment")

    args.path2save_train_history = config.path2save_train_history

    if not os.path.exists(args.path2save_train_history):
        try:
            os.mkdir(args.path2save_train_history)
        except:
            dir_path_head, dir_path_tail = os.path.split(args.path2save_train_history)
            if len(dir_path_tail) == 0:
                dir_path_head, dir_path_tail = os.path.split(dir_path_head)
            os.mkdir(dir_path_head)
            os.mkdir(args.path2save_train_history)

    main(args, args.is_evaluate)