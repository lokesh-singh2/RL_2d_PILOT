import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal


def GAE(rewards, values, next_values, terminated, done, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    for i in reversed(range(T)):
        delta = rewards[i] + gamma * next_values[i] * (1 - terminated[i]) - values[i]
        gae = delta + gamma * lam * (1 - done[i]) * gae
        advantages[i] = gae
    return advantages


def take_action(mu,std):
  dist=Normal(mu,std)
  u = dist.sample() # samples 2 actions each action for 1 motor
  log_prob=dist.log_prob(u).sum(-1)
  return u , log_prob


def is_crashed(next_state):
    return abs(next_state[3]) == 1 or abs(next_state[6]) == 1 or abs(next_state[7]) == 1


def compute_reward(next_state, crashed):
    # defining reward function  ########
    if crashed:
        reward = -10.0
    else:
        dx = next_state[4]
        dy = next_state[5]
        distance = np.sqrt(dx**2 + dy**2)
        pitch_angle_radians = next_state[3] * (np.pi / 2.0)
        reward = np.exp(-distance) * 0.5 + np.cos(pitch_angle_radians) * 0.5
    return reward


def plot_rewards(epoch_rewards_history, epochs, window_size=50, save_to=None):
    plt.figure(figsize=(12, 6))

    plt.plot(range(1, epochs + 1), epoch_rewards_history, label='Total Reward', color='blue', alpha=0.3)

    # Calculate and plot the moving average
    if len(epoch_rewards_history) >= window_size:
        # Calculate moving average using numpy
        moving_avg = np.convolve(epoch_rewards_history, np.ones(window_size)/window_size, mode='valid')
        # The moving average array is smaller by (window_size - 1), so we adjust the x-axis
        plt.plot(range(window_size, epochs + 1), moving_avg, color='red', linewidth=2)

    plt.xlabel('Epochs')
    plt.ylabel('Total Reward per Epoch')
    plt.title('PPO Agent Training Performance')
    plt.grid(True)
    plt.legend()
    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches='tight')
    plt.show()
