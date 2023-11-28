import torch
import numpy as np
from collections import deque

from Olympic.actor import ContinuousActor
from Olympic.actor_utils import init_weights

class running_2_agent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.actor = ContinuousActor(torch.device("cpu")).apply(init_weights).to(torch.device("cpu"))

        self.actor.load_state_dict(torch.load(
            "/Users/jang-gihwan/Desktop/DeepLearing/ReinforcementLearning/train_history/olympics-running/11_27_14_3/actor.pth",
            map_location=self.device))

        self.frame_stack = 4
        self.frames_controlled = deque([], maxlen=self.frame_stack)
        
    def act(self, obs):
        obs = self.obs_preprocessing(obs['obs']['agent_obs'])
        obs = np.reshape(obs, (1, 4, 40, 40)).astype(np.float64)
        obs = torch.FloatTensor(obs).to(self.device)
        action, _ = self.actor(obs)
        action = self.get_scaled_action(action)
        return action
    
    def obs_preprocessing(self, obs):
        obs = np.expand_dims(obs, axis=0)

        if len(self.frames_controlled) != self.frame_stack:
            for _ in range(self.frame_stack):
                self.frames_controlled.append(obs)
        else:
            self.frames_controlled.append(obs)
        obs = np.concatenate(list(self.frames_controlled), axis=0)
        obs = self.obs_convert(obs)

        return [obs]
    
    def obs_convert(self, obs):
        if obs[0][32][19] == 10:
            obs[np.where(obs == 8)] = 10
            obs[np.where(obs == 10)] = 8
        return obs
    
    def get_scaled_action(self, action):
        clipped_action = np.clip(action[0], -1.0, 1.0)

        scaled_action_0 = -30. + (clipped_action[0] + 1) * 115
        scaled_action_1 = -4 + (clipped_action[1] + 1) / 2 * (4 - (-4))
        return np.asarray([[scaled_action_0], [scaled_action_1]])