import numpy as np

import torch
torch.autograd.set_detect_anomaly(True)
from torch.optim import Adam
from torch import nn, Tensor
import torchvision.transforms as T

import gymnasium as gym
from typing import Tuple
import os
import pickle
import cv2

from ReplayBuffer import ReplayBuffer
from DQN import Network

class Agent:
    def __init__(
        self,
        env: gym.Env,
        chkpt_dir: str,
        gamma: float = 0.99,
        learning_rate: float = 0.0005,
        batch_size: int = 32,
        buffer_capacity: int = 1_000_000,
        obs_shape: Tuple[int, int] = (84, 84),
    ):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.obs_shape = obs_shape

        # Adding 1 to specify the input as grayscale
        self.net = Network((*obs_shape, 1), self.env.action_space.n)
        self.target_net = Network((*obs_shape, 1), self.env.action_space.n)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.optim = Adam(self.net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.update_parameters()

        chkpt_dir = chkpt_dir
        if not os.path.isdir(chkpt_dir):
            os.mkdir(chkpt_dir)
        
        # Create a buffer folder in checkpoint dir if not exists
        self.chkpt_dir_buffer = os.path.join(chkpt_dir, "buffer")
        if not os.path.isdir(self.chkpt_dir_buffer):
            os.mkdir(self.chkpt_dir_buffer)

        # Create a model folder in checkpoint dir if not exists
        self.chkpt_dir_models = os.path.join(chkpt_dir, "model")
        if not os.path.isdir(self.chkpt_dir_models):
            os.mkdir(self.chkpt_dir_models)

    
    def update_parameters(self, tau:float = 1.0):
        if tau is None:
            tau = 1.0
        
        target_state_dict = self.target_net.state_dict()
        policy_state_dict = self.net.state_dict()

        for (keys_t, keys_p) in zip(target_state_dict.keys(), policy_state_dict.keys()):
            target_state_dict[keys_t] = tau*policy_state_dict[keys_p] + (1-tau)*target_state_dict[keys_t]
        
        self.target_net.load_state_dict(target_state_dict)
    
    def preprocess(self, state : np.ndarray) -> Tensor:
        state = state[30:195]
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, self.obs_shape)

        # Add 1 dimension to change the state shape to [C, H, W]
        state = np.expand_dims(state, axis=0)
        return state
    
    def step(self, state: np.ndarray) -> Tuple:
        # Take the main portion of screen
        state = self.preprocess(state)

        # Convert to tensor and feed it to network
        q_vals = self.net(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        )

        action = np.argmax(q_vals.data.numpy())
        next_state, reward, done, _, info = self.env.step(action)

        # We do not want to store an extra dimension
        return (state, action, reward, done, next_state, info)

    def store(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            done: bool,
            next_state: np.ndarray
    ) -> None:
        self.buffer.append_experience(state, action, reward, done, next_state)
    
    def learn(self):
        if self.buffer.getlen() < self.batch_size:
            return

        state, action, reward, done, next_state = self.buffer.sample(self.batch_size)

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        
        state_action_val = self.net(state).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_state_actions = self.net(next_state).max(1)[1]
            next_state_action_val = self.target_net(next_state).gather(1, next_state_actions.unsqueeze(1)).squeeze(1)
            next_state_action_val[done] = 0.0

        expected_state_action_val = (next_state_action_val * self.gamma) + reward

        loss = self.criterion(expected_state_action_val, state_action_val)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    def save(self):
        torch.save(
            self.net.state_dict(),
            os.path.join(self.chkpt_dir_models, "net.pth")
        )
        torch.save(
            self.target_net.state_dict(), 
            os.path.join(self.chkpt_dir_models, "target_net.pth")
        )

        with open(os.path.join(self.chkpt_dir_buffer, "buffer.obj"), "wb") as buffer_file:
            pickle.dump(self.buffer, buffer_file)
        
        print("<======| Data Saved |======>")