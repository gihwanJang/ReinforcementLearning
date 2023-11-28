import sys
from pathlib import Path

base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)
print(sys.path)
import argparse
from termproject_olympic.olympics_engine.agent import *
import time
from termproject_olympic.env.chooseenv import make
import json

from ReinforcementLearning.Olympic.running_2_agent import running_2_agent


def store(record, name):

    with open('logs/'+name+'.json', 'w') as f:
        f.write(json.dumps(record))

def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson

RENDER = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", default="running")
    # "CartPole-v1", "Pendulum-v1", "Acrobot-v1", "LunarLanderContinuous-v2"
    # "LunarLander-v2", "BipedalWalker-v3", "MountainCarContinuous-v0"
    args, rest_args = parser.parse_known_args()
    env_name = args.env_type

    frame_skipping = 3
    frame_skipping_num = 0
    prev_action = None
    for i in range(1):
        if env_name == 'running':
            game = make(env_type="olympics-integrated", game_name='olympics-running')
            agent_num = 2
        elif env_name == 'wrestling':
            game = make(env_type="olympics-integrated", game_name='olympics-wrestling')
            agent_num = 2

        rand_agent = random_agent()

        ## chane yours agent ##
        test_agent = running_2_agent()

        obs = game.reset()
        done = False
        step = 0
        if RENDER:
            game.env_core.render()

        time_epi_s = time.time()
        while not done:
            step += 1

            if agent_num == 2:
                action1 = rand_agent.act(obs[0])
                action2 = test_agent.act(obs[1])
                action = [action1, action2]

            obs, reward, done, _, _ = game.step(action)
            print(f'reward = {reward}')
            # print('obs = ', obs)
            # plt.imshow(obs[0])
            # plt.show()
            if RENDER:
                game.env_core.render()

        duration_t = time.time() - time_epi_s
        print("episode duration: ", duration_t,
              "step: ", step,
              "time-per-step:",(duration_t)/step)
        # if args.map == 'billiard':
        #     print('reward =', game.total_reward)
        # else:
            # print('reward = ', reward)
        # if R:
        #     store(record,'bug1')

