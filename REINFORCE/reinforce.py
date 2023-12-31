import torch
from torch import nn
from torch.optim import Adam

import gymnasium as gym
import numpy as np

def create_nn(obs_shape: int, n_actions: int) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(obs_shape, 64),
        nn.ReLU(),
        nn.Linear(64, n_actions),
        nn.Softmax(dim=-1)
    )

    return net

def training_loop(
        env: gym.Env,
        net: nn.Sequential,
        optim: torch.optim,
        max_steps: int,
        y: float = 0.99
) -> nn.Sequential:
    
    steps = 0
    eps_count = 0

    while steps < max_steps:
        # Initialize action, state, reward holders
        Actions, States, Rewards = [], [], []

        obs, info = env.reset()
        done = False

        obs = torch.tensor(obs, dtype=torch.float32)
        eps_count += 1

        while not done:
            # Obtain the probability
            prob = net(obs)
            dist = torch.distributions.Categorical(probs=prob)
            action = dist.sample().item()

            next_obs, r, done, _, _ = env.step(action)

            Actions.append(torch.tensor(action, dtype=torch.int))
            States.append(obs)
            Rewards.append(r)

            obs = torch.tensor(next_obs, dtype=torch.float32)
            steps += 1
        
        # Once the episode is completed
        DiscountedReturns = []
        for t in range(len(Rewards)):
            G = 0
            for k, r in enumerate(Rewards[t:]):
                G += (y**k)*r
            
            DiscountedReturns.append(G)
        
        for s, a, G in zip(States, Actions, DiscountedReturns):
            prob = net(s)
            dist = torch.distributions.Categorical(probs=prob)
            log_prob = dist.log_prob(a)
            
            loss = -(log_prob * G)

            optim.zero_grad()
            loss.backward()
            optim.step()
        
        print(f"Episode: {eps_count} | Reward: {sum(Rewards):.2f}")

    return net

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="human")
    net = create_nn(env.observation_space.shape[0], env.action_space.n)
    
    learning_rate = 0.005

    optim = Adam(net.parameters(), lr=learning_rate)
    
    max_steps = 50_000
    y = 0.9999

    print("Starting Traning")
    net = training_loop(env, net, optim, max_steps, y)
    torch.save(net.state_dict(), "./net.pth")