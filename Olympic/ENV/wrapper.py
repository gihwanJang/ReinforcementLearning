import os
import sys
import gym
import numpy as np
import random

from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

class CompetitionOlympicsEnvWrapper(gym.Wrapper):
    metadata = {}

    def __init__(self, env, agent=None, args=None):
        super().__init__(env)

        self.args = args
        assert self.args

        self.controlled_agent_index = args.controlled_agent_index

        self.frame_stack = args.frame_stack
        assert self.frame_stack > 0 or isinstance(self.frame_stack, int)
        self.frames_controlled = deque([], maxlen=self.frame_stack)
        self.frames_opponent = deque([], maxlen=self.frame_stack)

        self.sub_game = args.env_name
        self.device = args.device

        self.episode_steps = 0
        self.total_steps = 0

    def reset(self):
        self.episode_steps = 0
        observation = self.env.reset()

        observation_opponent_agent = np.expand_dims(
            observation[1 - self.controlled_agent_index]['obs']['agent_obs'], axis=0
        )
        observation_controlled_agent = np.expand_dims(
            observation[self.controlled_agent_index]['obs']['agent_obs'], axis=0
        )

        observation_opponent_agent = self.frame_stacking(self.frames_opponent, observation_opponent_agent)
        observation_controlled_agent = self.frame_stacking(self.frames_controlled, observation_controlled_agent)

        return [observation_controlled_agent], None    

    def step(self, action_controlled):
        # if is_evaluate true -> do game using agent, no train
        if self.args.render_over_train or self.args.is_evaluate:
            self.render()

        self.episode_steps += 1
        self.total_steps += 1

        action_controlled = self.get_scaled_action(action_controlled)
        action_opponent = self.get_opponent_action()

        action_controlled = np.expand_dims(action_controlled, axis=1)
        action_opponent = np.expand_dims(action_opponent, axis=1)

        action = [action_opponent, action_controlled] if self.args.controlled_agent_index == 1 else [action_controlled, action_opponent]

        next_observation, reward, done, info_before, info_after = self.env.step(action)

        next_observation_opponent_agent = np.expand_dims(next_observation[1 - self.controlled_agent_index]['obs']['agent_obs'], axis=0)
        next_observation_controlled_agent = np.expand_dims(next_observation[self.controlled_agent_index]['obs']['agent_obs'], axis=0)

        self.frames_opponent.append(next_observation_opponent_agent)
        next_observation_opponent_agent = self._transform_observation(self.frames_opponent)

        self.frames_controlled.append(next_observation_controlled_agent)
        next_observation_controlled_agent = self._transform_observation(self.frames_controlled)

        reward_controlled = self._transform_reward(reward[self.controlled_agent_index], next_observation_controlled_agent)

        info = {}

        return [next_observation_controlled_agent], reward_controlled, done, False, info

    def render(self, mode='human'):
        self.env.env_core.render()

    def close(self):
        pass

    def get_opponent_action(self):
        force = random.uniform(-100, 200)
        angle = random.uniform(-30, 30)
        opponent_scaled_actions = np.asarray([force, angle])

        return opponent_scaled_actions

    # 주어진 action을 force : -100~100, angle : -30~30으로 변경 -> force를 -100~200으로 변화 시킬 필요 있어 보임  추후 변경 요망
    def get_scaled_action(self, action):
        clipped_action = np.clip(action, -1.0, 1.0)

        #scaled_action_0 = -100 + (clipped_action[0] + 1) / 2 * (200 - (-100))
        scaled_action_0 = -20 + (clipped_action[0] + 1) / 2 * (200 - 1)
        #scaled_action_1 = -30 + (clipped_action[1] + 1) / 2 * (30 - (-30))
        scaled_action_1 = -5 + (clipped_action[1] + 1) / 2 * (5 - (-5))

        return np.asarray([scaled_action_0, scaled_action_1])

    def frame_stacking(self, deque, obs):
        for _ in range(self.frame_stack):
            deque.append(obs)
        obs = self._transform_observation(deque)
        return obs

    def _transform_observation(self, frames):
        assert len(frames) == self.frame_stack
        obs = np.concatenate(list(frames), axis=0)
        return obs

    def _transform_reward(self, prev_reward, next_observation):
        center_x, center_y = 20, 20
        central_region = next_observation[0][center_x - 5:center_x + 5, center_y - 5:center_y + 5]
    
        # 중앙 영역에 7이 포함되어 있는지 확인
        new_reward = 0
        if 7 in central_region:
            new_reward += 1000000
        if 6 in central_region:
            new_reward += 1
    
        return new_reward