import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import drone_2d_custom_gym_env

from drone_ppo.models import Actor_NN
from drone_ppo.utils import is_crashed, compute_reward

checkpoint = 'checkpoints/ppo(2000).pth'   # which saved weights to test (best model)
episodes = 5                   # how many flights to run

env = gym.make(
    'drone-2d-custom-v0',
    disable_env_checker=True,
    render_sim=True,
    render_path=True,
    render_shade=True
)

actor = Actor_NN(features=8)
actor.load_state_dict(torch.load(checkpoint))
# actor.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))  # if no GPU
actor.eval()

print("Weights loaded successfully. Starting simulation...")

episode_rewards = []
crash_count = 0

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        try:
            env.render()
        except Exception:
            pass

        state_tensor = torch.FloatTensor(state)

        with torch.no_grad():
            mu, std = actor(state_tensor)
            action = torch.clamp(mu, -1, 1).numpy()   # deterministic: use the mean, no noise

        state, reward, done, info = env.step(action)

        crashed = is_crashed(state)
        reward = compute_reward(state, crashed)       # same reward as in training

        total_reward += reward
        step_count += 1

    if crashed:
        crash_count += 1

    episode_rewards.append(total_reward)
    print(f"Episode {ep + 1}/{episodes} | Steps Survived: {step_count} | Total Reward: {total_reward:.2f}")

env.close()

print(f"\nMean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
print(f"Crash rate : {100 * crash_count / episodes:.1f}%")

# ---------------- Plot: reward vs episode ----------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, episodes + 1), episode_rewards, marker='o')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward vs Episode')
plt.xticks(range(1, episodes + 1))
plt.grid(True)
plt.show()
