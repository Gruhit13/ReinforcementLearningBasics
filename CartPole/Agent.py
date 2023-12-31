import numpy as np
import torch
from torch import nn, optim
from DQN import DQN
from ReplayBuffer import ReplayBuffer
import gym

class Agent:
	def __init__(
		self,
		env_name,
		batch_size=64,
		gamma=0.99,
		eps_start = 1.0,
		eps_end = 0.05,
		eps_decay = 1000,
		sync_rate = 4,
		max_capacity = 5000,
		tau = 0.005,
		alpha = 1e-4
	):

		self.env = gym.make(env_name)

		self.batch_size = batch_size
		self.gamma = gamma
		self.eps_start = eps_start
		self.eps_end = eps_end
		self.eps_decay = eps_decay
		self.sync_rate = sync_rate
		self.tau = tau

		# This variable will count the total steps played so far
		self.steps_cnt = 0

		self.n_action = self.env.action_space.n

		self.replay_buffer = ReplayBuffer(max_capacity, self.env.observation_space.shape)

		self.net = DQN(self.env.observation_space.shape[0], self.n_action)
		self.target_net = DQN(self.env.observation_space.shape[0], self.n_action)

		self.optimizer = optim.Adam(self.net.parameters(), lr=alpha)
		self.criterion = nn.MSELoss()
		self.update_parameters()

	def update_parameters(self, tau=None):

		if tau is None:
			tau = 1.

		net_state_dict = self.net.state_dict()
		target_net_state_dict = self.target_net.state_dict()

		for key in net_state_dict.keys():
			target_net_state_dict[key] = (net_state_dict[key]*tau) + \
										 (target_net_state_dict[key]*(1 - tau))

		self.target_net.load_state_dict(target_net_state_dict)

	def get_action(self, state, eps):
		if np.random.random() < eps:
			return np.random.choice(self.n_action)

		else:
			self.net.eval()
			state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

			with torch.no_grad():
				action_value = self.net(state)

			self.net.train()
			return np.argmax(action_value.data.numpy())


	def get_epsilon(self):
		if self.steps_cnt > self.eps_decay:
			return self.eps_end
		else:
			return self.eps_start - (self.steps_cnt/self.eps_decay)*(self.eps_start - self.eps_end)

	def play_step(self, state):
		epsilon = self.get_epsilon()
		action = self.get_action(state, epsilon)

		next_state, reward, done, terminal, info = self.env.step(action)

		self.steps_cnt += 1

		return action, next_state, reward, done

	def append(self, state, action, reward, done, next_state):
		self.replay_buffer.store_experience(state, action, reward, done, next_state)

	def learn(self):

		if self.batch_size > self.replay_buffer.mem_size:
			return

		state, action, reward, done, next_state = self.replay_buffer.sample(self.batch_size)

		state = torch.tensor(state, dtype=torch.float32)
		action = torch.tensor(action, dtype=torch.int64)
		reward = torch.tensor(reward, dtype=torch.float32)
		done = torch.tensor(done, dtype=torch.float32)
		next_state = torch.tensor(next_state, dtype=torch.float32)
		
		self.net.train()
		self.target_net.eval()

		state_action_value = self.net(state).gather(1, action.unsqueeze(1))

		with torch.no_grad():
			next_state_action_value = self.target_net(next_state).detach().max(1)[0]

		expected_state_action_value = reward + (self.gamma * next_state_action_value * (1 - done))
		expected_state_action_value = expected_state_action_value.unsqueeze(1)

		loss = self.criterion(state_action_value, expected_state_action_value)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()