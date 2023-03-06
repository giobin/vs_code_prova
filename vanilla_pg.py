import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from gymnasium.wrappers import RecordVideo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-episods", type=int, default=1000,
        help="total training episods")
    parser.add_argument("--max-episod-steps", type=int, default=500,
        help="maximum number of steps per episod")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    
    args = parser.parse_args()
    return args

class Agent(nn.Module):
    def __init__(self,env) -> None:
        super().__init__()
        self.lin1 = nn.Linear(env.observation_space.shape[0], 64)
        self.lin2 = nn.Linear(64, env.action_space.n)

    def forward(self, x):
        actions = self.lin2(torch.relu(self.lin1(x)))
        return torch.softmax(actions, dim=-1)
    
    def sample(self, x):
        probs = self.forward(x)
        distrib = Categorical(probs)
        action = distrib.sample()
        log_prob = distrib.log_prob(action).unsqueeze(0)
        return action, log_prob
    
    def compute_vanilla_PG_loss(self, buffer):
        returns = buffer.compute_discounted_rewards()
        returns = torch.tensor(returns)
        log_probs = torch.cat(buffer.log_probs)
        loss = torch.sum(-log_probs * returns)
        return loss

    
class Buffer():
    def __init__(self) -> None:
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
    
    def compute_discounted_rewards(self):
        R = 0
        returns = deque()
        rewards = np.array(self.rewards)
        #scaled_rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for r in rewards[::-1]:
            R = r + 0.99 * R
            returns.appendleft(R)
        return returns
    
    def add_exp(self, observation, action, log_probs, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        return

def evaluate(episods_num, episod_steps, agent, env):
    observation, info = env.reset(seed=30)
    episod_rewards = []

    for e in range(episods_num):
        episod_reward = 0 
        for s in range(episod_steps):
            
            observation = torch.from_numpy(observation)
            action = torch.argmax(agent(observation))
            observation, reward, terminated, truncated, info = env.step(action.item())

            episod_reward += reward

            if terminated or truncated:
                episod_rewards.append(episod_reward)

                #print(f"episod: {e}, step: {s}, loss: {loss.item()}, episod_reward: {episod_reward}, mean_episod_reward: {mean_episod_rewards}")
                if e % 10 == 0: print(f"episod: {e}, step: {s}, reward = {episod_reward}")

                observation, info = env.reset()
                break

    mean_episod_rewards = np.array(episod_rewards).mean()
    mean_episod_std = np.array(episod_rewards).std()
    return mean_episod_rewards, mean_episod_std


def main(args):

    run_name = f"{args.env_id}__{args.exp_name}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    env = gym.make("CartPole-v1")
    agent = Agent(env=env)
    optimizer = optim.Adam(agent.parameters())

    observation, info = env.reset(seed=42)

    episod_rewards, losses = [], []

    for e in range(args.total_episods):
        buffer = Buffer()
        episod_reward = 0 

        for s in range(args.max_episod_steps):
            
            observation = torch.from_numpy(observation)
            action, log_probs = agent.sample(observation)
            observation, reward, terminated, truncated, info = env.step(action.item())

            # save the observation and reward
            buffer.add_exp(observation, action, log_probs, reward)
            episod_reward += reward

            if terminated or truncated:
                # compute the loss
                loss = agent.compute_vanilla_PG_loss(buffer)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episod_rewards.append(episod_reward)
                losses.append(loss.item())
                # the mean_episod_rewards is just the average reward of the last 10 episodes
                mean_episod_rewards = np.array(episod_rewards[-10:]).mean()
                mean_losses = np.array(losses[-10:]).mean()

                #print(f"episod: {e}, step: {s}, loss: {loss.item()}, episod_reward: {episod_reward}, mean_episod_reward: {mean_episod_rewards}")
                if e % 100 == 0: print(f"episod {e}-{s} -> mean-episod-reward: {mean_episod_rewards}, mean-loss: {mean_losses}") 
                writer.add_scalar("mean_episod_rewards", mean_episod_rewards, s*e)
                writer.add_scalar("mean_losses", mean_losses, s*e)

                observation, info = env.reset()
                break

    env.close()

    env = RecordVideo(gym.make("CartPole-v1", render_mode="rgb_array"), video_folder="./video-cartpole-v1/", name_prefix=run_name)
    mean_episod_rewards, mean_episod_std = evaluate(100, args.max_episod_steps, agent, env)
    print(f"mean_episod_rewards: {mean_episod_rewards}, mean_episod_std: {mean_episod_std}")

if __name__ == "__main__":
    args = parse_args()

    main(args)