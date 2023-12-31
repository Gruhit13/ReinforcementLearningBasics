from ReplayBuffer import ReplayBuffer
from Network import Network
from typing import Tuple
import os
import gymnasium as gym
import numpy as np
import pickle

import torch
from torch import nn
from torch.optim import Adam

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(
        self,
        env: str,
        chkpt_dir: str,
        eps_start: float = 1.0,
        eps_final: float = 0.01,
        eps_decay_last_frame: int = 10**5,
        gamma: float = 0.99,
        learning_rate: float = 0.005,
        batch_size: int = 32,
        buffer_capacity: int = 1_000_000,
        obs_shape: Tuple[int, int] = (84, 84),
    ):  
        self.env = gym.make(env)
        self.env = gym.wrappers.AtariPreprocessing(
            self.env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False
        )

        self.gamma = gamma
        self.batch_size = batch_size
        self.obs_shape = obs_shape
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay_last_frame = eps_decay_last_frame

        # Adding 1 to specify the input as grayscale
        self.net = Network((*obs_shape, 1), self.env.action_space.n).to(device)
        self.target_net = Network((*obs_shape, 1), self.env.action_space.n).to(device)

        # In buffer we need to pass [C, H, W] shape for image
        self.buffer = ReplayBuffer(buffer_capacity, obs_shape)
        self.optim = Adam(self.net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

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

        self.update_parameters()

    def update_parameters(self, tau:float = 1.0):

        self.target_net.load_state_dict(
            self.net.state_dict()
        )
    
    def calculate_eps(self, frame_idx) -> float:
        return max(self.eps_final, self.eps_start - frame_idx / self.eps_decay_last_frame)

    def get_action(self, state, frame_idx):
        eps = self.calculate_eps(frame_idx)

        if np.random.uniform() < eps:
            action = self.env.action_space.sample()
        else:
            inp_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            # Convert to tensor and feed it to network
            q_vals = self.net(
                inp_state
            )

            action = np.argmax(q_vals.detach().cpu().numpy())
        
        return action, eps

    def step(self, state: np.ndarray, frame_idx: int) -> Tuple:

        action, eps = self.get_action(state, frame_idx)
        
        next_state, reward, done, _, info = self.env.step(action)
        next_state = np.expand_dims(next_state, axis=0)

        # We do not want to store an extra dimension
        return (state, action, reward, done, next_state, info, eps)

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

        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.int64).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

        state_action_val = self.net(state).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_state_action_val = self.target_net(next_state).max(1)[0]

        expected_state_action_val = (next_state_action_val * self.gamma * (1 - done)) + reward

        loss = self.criterion(expected_state_action_val, state_action_val).to(device)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.net.reset_noise()
        self.target_net.reset_noise()

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