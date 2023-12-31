import numpy as np
from collections import deque
from typing import Tuple

class ReplayBuffer:
    def __init__(
            self, 
            max_capacity: int,
        ) -> None:
        self.buffer = deque(maxlen=max_capacity)
    
    def append_experience(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            done: bool,
            next_state: np.ndarray
    ) -> None:
        self.buffer.append((
            state,
            action,
            reward,
            done,
            next_state
        ))
    
    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        states, actions, rewards, dones, next_states = zip(*(self.buffer[i] for i in indices ))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(dones),
            np.array(next_states)
        )

    def getlen(self):
        return len(self.buffer)