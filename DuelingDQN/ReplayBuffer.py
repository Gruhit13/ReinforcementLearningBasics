import numpy as np
from typing import Tuple

class ReplayBuffer:
    def __init__(
            self,
            max_capacity: int,
            obs_shape: Tuple[int, int]
        ) -> None:

        self.maxlen = max_capacity

        self.state = np.empty((self.maxlen, 1, *obs_shape)).astype('uint8')
        self.action = np.empty((self.maxlen, ), dtype=np.int32)
        self.reward = np.empty((self.maxlen, ), dtype=np.float32)
        self.done = np.empty((self.maxlen, ), dtype=bool)
        self.next_state = np.empty((self.maxlen, 1, *obs_shape)).astype('uint8')

        # self.buffer = deque(maxlen=max_capacity)
        self.cntr = 0
        self.mem_size = 0

    def append_experience(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            done: bool,
            next_state: np.ndarray
    ) -> None:
        state = state.astype('uint8')
        next_state = next_state.astype('uint8')

        self.state[self.cntr] = state
        self.action[self.cntr] = action
        self.reward[self.cntr] = reward
        self.done[self.cntr] = done
        self.next_state[self.cntr] = next_state

        self.cntr = (self.cntr + 1) % self.maxlen
        self.mem_size = min(self.mem_size+1, self.maxlen)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(self.mem_size, batch_size, replace=False)

        states = self.state[indices]
        actions = self.action[indices]
        rewards = self.reward[indices]
        dones = self.done[indices]
        next_states = self.next_state[indices]

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(dones),
            np.array(next_states)
        )

    def getlen(self):
        return self.mem_size