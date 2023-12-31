import numpy as np

class ReplayBuffer:
	def __init__(self, capacity, obs_shape):

		self.state = np.zeros((capacity, *obs_shape), dtype=np.float32)

		self.action = np.zeros((capacity), dtype=np.float32)
		self.reward = np.zeros((capacity), dtype=np.float32)
		self.done = np.zeros((capacity), dtype=bool)
		self.next_state = np.zeros((capacity, *obs_shape), dtype=np.float32)

		self.max_size = capacity
		self.mem_cnt = 0
		self.mem_size = 0

	def store_experience(self, state, action, reward, done, next_state):

		self.state[self.mem_cnt] = state
		self.action[self.mem_cnt] = action
		self.reward[self.mem_cnt] = reward
		self.done[self.mem_cnt] = done
		self.next_state[self.mem_cnt] = next_state

		self.mem_cnt = (self.mem_cnt + 1) % self.max_size
		self.mem_size = min(self.mem_size+1, self.max_size)

	def sample(self, batch_size):
		if batch_size > self.mem_size:
			return
		
		idx = np.random.randint(0, self.mem_size, batch_size)

		return self.state[idx], \
			   self.action[idx], \
			   self.reward[idx], \
			   self.done[idx], \
			   self.next_state[idx]