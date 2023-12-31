import gymnasium as gym
import numpy as np
from typing import Union, Dict

from Agent import Agent
from config import CONFIG

def reset(env : gym.Env) -> Union[np.ndarray, Dict[str, int]]:
    """
    Custom reset environment as Atari environment does not work for
    first 85 frames and hence it is useless to store that frames
    """

    state, info = env.reset()
    for _ in range(85):
        act = env.action_space.sample()
        next_state, _, _, _, info = env.step(act)
    
    return next_state, info


def play():
    env = gym.make(CONFIG["ENV_NAME"])

    state, info = env.reset()
    episode = 0
        
    agent = Agent(
        env=env,
        chkpt_dir=CONFIG['CHKPT_DIR'],
        gamma=CONFIG['GAMMA'],
        learning_rate=CONFIG['LEARNING_RATE'],
        batch_size=CONFIG['BATCH_SIZE'],
        buffer_capacity=CONFIG['REPLAY_SIZE'],
        obs_shape=CONFIG['OBS_SHAPE']
    )
    
    rewards = []

    best_reward = 0

    frame_cnt = 0

    while True:
        episode += 1

        done = False
        state, info = reset(env)
        lives = info['lives']
        eps_reward = 0

        while not done:
            # State obtained here is processed but next_state is not processed
            state, act, reward, done, next_state, info = agent.step(state)
            frame_cnt += 1
            
            if info['lives'] < lives:
                done = True
            
            # Preprocess next_state before passing it to the store
            agent.store(state, act, reward, done, agent.preprocess(next_state))

            # Copy the original next_state over state to get prediction in next frame
            state = next_state.copy()
            eps_reward += reward

            agent.learn()

            if frame_cnt % CONFIG['SYNC_RATE'] == 0:
                agent.update_parameters(CONFIG['TAU'])
        
        rewards.append(eps_reward)
        avg_reward = np.mean(rewards[-100:])

        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save()
        
        print(f"{episode} | Episodic Reward: {eps_reward:.2f} | Avg Reward: {avg_reward:2f} | Best Reward: {best_reward:2f} | Frame Cnt: {frame_cnt}")

        if True: break

        if frame_cnt >= CONFIG['MAX_FRAME']:
            break

if __name__ == "__main__":
    play()
        
