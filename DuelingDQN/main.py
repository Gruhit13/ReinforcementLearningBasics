import numpy as np
from Agent import Agent

CONFIG = {
    "MAX_FRAME": 1_000_000,
    "MIN_FRAME_TO_WAIT": 1_000,
    "SYNC_RATE": 1000,
    "REPLAY_SIZE": 100_000,
    "LEARNING_RATE": 0.0001,
    "GAMMA": 0.99,
    "BATCH_SIZE": 32,
    "TAU": 1.0,
    "ENV_NAME": "PongNoFrameskip-v4",
    "OBS_SHAPE": (84, 84),
    "CHKPT_DIR": "./checkpoint"
}

def play(agent):
    episode = 0

    rewards = []
    best_reward = 0
    frame_cnt = 0

    while True:
        episode += 1

        done = False

        eps_reward = 0
        eps_frame_cnt = 0

        state, info = agent.env.reset()
        state = np.expand_dims(state, axis=0)

        while not done:
            # State obtained here is processed but next_state is not processed
            state, act, reward, done, next_state, info, eps = agent.step(state)
            frame_cnt += 1
            eps_frame_cnt += 1

            # Preprocess next_state before passing it to the store
            agent.store(state, act, reward, done, next_state)

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

        print("%d | Episodic Reward: %.2f | Avg Reward: %.2f | "\
              "Best Reward: %.2f | Eps Frame Cnt: %d | Frame Count: %d "\
              "Epsilon: %.2f"\
              %(episode, eps_reward, avg_reward, best_reward, eps_frame_cnt, frame_cnt))

        if frame_cnt >= CONFIG['MAX_FRAME']:
            break

    return rewards