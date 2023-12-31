import numpy as np
import gym
import torch
from Agent import Agent

def main():
	env_name = "CartPole-v1"
	
	EPISODE = 600
	# Maximum frame an episode can last
	MAX_EPISODE_LEN = 500
	episodic_rewards = []
	best_reward = 0

	agent = Agent(env_name)

	for i in range(EPISODE):
		done = False
		state, _ = agent.env.reset()
		episodic_reward = 0

		eps_frame_cnt = 0

		while not done:
			action, next_state, reward, done = agent.play_step(state)
			eps_frame_cnt += 1

			episodic_reward += reward

			agent.append(state, action, reward, done, next_state)

			# Learn on every move. i.e. imrpovise sampling policy every move
			agent.learn()
			
			# Improve the target policy after N moves
			if agent.steps_cnt % agent.sync_rate == 0:
				agent.update_parameters()

			state = next_state

			if eps_frame_cnt % MAX_EPISODE_LEN == 0:
				break

		episodic_rewards.append(episodic_reward)
		avg_reward = np.mean(episodic_rewards[:-10])

		if avg_reward > best_reward:
			model_script = torch.jit.script(agent.net)
			model_script.save("./best_model.pt")
			best_reward = avg_reward

		print(f"Episode: {i+1} | Episodic_reward: {episodic_reward} | Best Reward: {best_reward:.2f}")

if __name__ == "__main__":
	main()