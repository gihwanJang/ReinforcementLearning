import os
import sys
import time
import torch
import gym
import numpy as np
import random

from collections import deque
from actor import ContinuousActor
from actor_utils import init_weights

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

class CompetitionOlympicsRunningEnvWrapper(gym.Wrapper):
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
        # 상대 에이전트
        self.opponent_agent = ContinuousActor(torch.device("cpu")).apply(init_weights).to(torch.device("cpu"))
        # 상대 에이전트
        self.opponent_agent.load_state_dict(torch.load(
            "/Users/jang-gihwan/Desktop/DeepLearing/ReinforcementLearning/train_history/olympics-running/11_27_21_41/actor.pth",
            map_location=self.device))

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
        # 상대 에이전트
        self.observation_opponent_agent = [observation_opponent_agent]
        
        return [observation_controlled_agent], None    

    def step(self, action_controlled):
        if self.args.is_evaluate:
            time.sleep(0.05)
            
        if self.args.render_over_train or self.args.is_evaluate:
            self.render()

        self.episode_steps += 1
        self.total_steps += 1

        action_controlled = self.get_scaled_action(action_controlled)
        
        # 상대 에이전트
        state = np.array(self.observation_opponent_agent)
        state = torch.FloatTensor(state).to(self.device)
        action_opponent, _ = self.opponent_agent(state)
        # 상대 에이전트
        action_opponent = self.get_scaled_action(list(action_opponent.detach().cpu().numpy()).pop())

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
        # 상대 에이전트
        self.observation_opponent_agent = [next_observation_opponent_agent]

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

    def get_scaled_action(self, action):
        clipped_action = np.clip(action, -1.0, 1.0)

        scaled_action_0 = -32. + (clipped_action[0] + 1) * 115
        scaled_action_1 = -4. + (clipped_action[1] + 1) / 2 * (4 - (-4))
        
        return np.asarray([scaled_action_0, scaled_action_1])

    def frame_stacking(self, deque, obs):
        for _ in range(self.frame_stack):
            deque.append(obs)
        obs = self._transform_observation(deque)
        return obs

    def _transform_observation(self, frames):
        assert len(frames) == self.frame_stack
        obs = np.concatenate(list(frames), axis=0)
        
        if self.controlled_agent_index == 0 :
            obs[np.where(obs == 8)] = 10
            obs[np.where(obs == 10)] = 8
        
        return obs

    def _transform_reward(self, prev_reward, next_observation):
        center_x, center_y = 32, 19
        new_reward = 0
        
        for i in range(4):
            central_region = next_observation[i][center_x - 5:center_x + 5, center_y - 5:center_y + 5]
            central_region_for_wall = next_observation[i][center_x - 10:center_x + 10, center_y - 10:center_y + 10]
            
            if 6 in central_region_for_wall:
                new_reward -= 6
            if 10 in central_region:
                new_reward -= 2
            if 0 in central_region:
                new_reward -= 1
        # if 4 in central_region:
        #     new_reward += 1
    
        return (prev_reward * 10000) + new_reward
    
    
class CompetitionOlympicsWrestlingEnvWrapper(gym.Wrapper):
    def __init__(self, env, agent=None, args=None):
        super().__init__(env)

        self.args = args
        assert self.args

        self.controlled_agent_index = args.controlled_agent_index

        # for frame_stack
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

        # print(f"{self.sub_game = }")
        # print(observation_controlled_agent)
        # print(f"before frame stacking!! {observation_controlled_agent.shape = }")

        ######### frame stack #########
        observation_opponent_agent = self.frame_stacking(self.frames_opponent, observation_opponent_agent)
        observation_controlled_agent = self.frame_stacking(self.frames_controlled, observation_controlled_agent)

        # print(f"after frame stacking!! {observation_controlled_agent.shape = }")
        ################################

        return [observation_controlled_agent], None

    def step(self, action_controlled):

        if self.args.render_over_train or self.args.is_evaluate:
            self.render()

        self.episode_steps += 1
        self.total_steps += 1

        action_controlled = self.get_scaled_action(action_controlled)
        action_opponent = self.get_opponent_action()
        # print(action_controlled, f"{action_controlled.shape = }")

        action_controlled = np.expand_dims(action_controlled, axis=1)
        action_opponent = np.expand_dims(action_opponent, axis=1)
        # print(f"after expanding dims!! {action_controlled}, {action_controlled.shape = }")

        action = [action_opponent, action_controlled] if self.args.controlled_agent_index == 1 else [
            action_controlled, action_opponent]
        # print(f"final action!! {action}")

        next_observation, reward, done, info_before, info_after = self.env.step(action)

        next_observation_opponent_agent = np.expand_dims(
            next_observation[1 - self.controlled_agent_index]['obs']['agent_obs'], axis=0)
        next_observation_controlled_agent = np.expand_dims(
            next_observation[self.controlled_agent_index]['obs']['agent_obs'], axis=0
        )

        ######### frame stack #########
        self.frames_opponent.append(next_observation_opponent_agent)
        next_observation_opponent_agent = self._transform_observation(self.frames_opponent)

        self.frames_controlled.append(next_observation_controlled_agent)
        next_observation_controlled_agent = self._transform_observation(self.frames_controlled)
        ################################

        reward_controlled = reward[self.controlled_agent_index]
        
        reward_controlled = self._transform_reward(reward[self.controlled_agent_index], next_observation_controlled_agent)

        info = {}

        return [next_observation_controlled_agent], reward_controlled, done, False, info

    def render(self, mode='human'):
        self.env.env_core.render()

    def close(self):
        pass

    def get_opponent_action(self):
        force = random.uniform(-100, 200) * 0
        angle = random.uniform(-30, 30) * 0
        opponent_scaled_actions = np.asarray([force, angle])

        return opponent_scaled_actions

    def get_scaled_action(self, action):
        clipped_action = np.clip(action, -1.0, 1.0)

        scaled_action_0 = -30. + (clipped_action[0] + 1) * 30
        scaled_action_1 = -5. + (clipped_action[1] + 1) / 2 * (5 - (-5))
        
        return np.asarray([scaled_action_0, scaled_action_1])

    def frame_stacking(self, deque, obs):
        for _ in range(self.frame_stack):
            deque.append(obs)
        obs = self._transform_observation(deque)
        return obs

    def _transform_observation(self, frames):
        assert len(frames) == self.frame_stack
        obs = np.concatenate(list(frames), axis=0)
        
        if self.controlled_agent_index == 0 :
            obs[np.where(obs == 8)] = 10
            obs[np.where(obs == 10)] = 8
        
        return obs

    def _transform_reward(self, prev_reward, next_observation):
        center_x, center_y = 32, 19
        new_reward = 0
        
        for i in range(4):
            central_region_for_wall = next_observation[i][center_x - 10:center_x + 10, center_y - 10:center_y + 10]
            
            self._fill_outter(next_observation, i)
                
            for r in range(18):
                for c in range(18):
                    new_reward += central_region_for_wall[r][c]
                    
        #print(next_observation)
        
        if prev_reward < 0:
            new_reward -= 1000000
        return new_reward
    
    def _fill_outter(self, next_observation, i):
        outter = deque()
        one_idx = None
        four_idx = None
        # 가로
        for r in range(40):
            for c in range(40):
                if next_observation[i][r][c] == 1 :
                    one_idx = [r,c]
                elif next_observation[i][r][c] == 4 :
                    four_idx = [r,c]
                    
        if one_idx != None and four_idx != None:                    
            outter.append([int((one_idx[0] + four_idx[0]) / 2), int((one_idx[1] + four_idx[1]) / 2)])
            self._BFS(next_observation, i, outter)
                        
    def _BFS(self, next_observation, i, outter):
        dx = [-1,1,0,0]
        dy = [0,0,-1,1]
        
        while outter:
            curr = outter.pop()
            
            if next_observation[i][curr[0]][curr[1]] == 0.:
                next_observation[i][curr[0]][curr[1]] = -100
                
                for i in range(4):
                    next_r = curr[0] + dx[i]
                    next_c = curr[1] + dy[i]
                    
                    if self._is_validate(next_observation[i], next_r, next_c):
                        outter.append([next_r, next_c])
                    
                    
    def _is_validate(self,next_observation, r, c):
        return ( (0 <= r < 40 and 0 <= c < 40)
                 and
                 (next_observation[r][c] == 0.)
                )
                
                