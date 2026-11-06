import torch
import gym
from torch.distributions import Normal
import numpy as np
import drone_2d_custom_gym_env

from drone_ppo.models import Actor_NN, Critic_NN
from drone_ppo.utils import GAE, take_action, is_crashed, compute_reward, plot_rewards
from drone_ppo.config import (gamma, lam, eps, learning_rate, ent_coef, epochs, N, batch_size,
                    update_epochs, max_grad_norm, window_size, save_path)

env = gym.make('drone-2d-custom-v0', disable_env_checker=True )


################################################################################


actor=Actor_NN(8)
critic=Critic_NN(8)

a_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
c_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)

D = []  # buffer to store the collected data
epoch_rewards_history = []


######################################################################


for epoch in range(epochs):
    
    state  = env.reset()
    done = False

    current_epoch_reward = 0

    if epoch % 50 == 0:
      print(f"current epoch is {epoch}")

    while not done:

        state = torch.FloatTensor(state)

        # print(state.shape)

        with torch.no_grad():
            mu, std = actor(state)
            action, log_prob = take_action(mu, std)
            s_action = action
            action = torch.clamp(action, -1, 1)
            action=action.numpy()

        next_state, reward, done , info = env.step(action)

        crashed = is_crashed(next_state)

        reward = compute_reward(next_state, crashed)

        current_epoch_reward += reward


        truncated = done and not crashed
        terminated = done and crashed


        # print(info)

        D.append((state, s_action, reward, next_state, log_prob, float(terminated), float(done)))

        ###################################################################

        if len(D) == N:

            (collected_state, collected_action, collected_reward, collected_next_state,collected_log_prob, collected_terminated, collected_done) = zip(*D)

            collected_state = torch.stack(collected_state)
            collected_action = torch.stack(collected_action)
            collected_log_prob = torch.stack(collected_log_prob)
            
            #collected_state = (tensor_0, tensor_1, ..., tensor_2047)
            # torch.stack	Adds a new dimension. (8,) × 2048 → (2048, 8)
            # It is a tuple, not a tensor. It holds 2048 small tensors.The networks cannot take a tuple. They need one tensor
            
            #above 3 are already tensors, below 4 are lists of floats, so we need to convert them to tensors

            collected_reward = torch.tensor(np.array(collected_reward), dtype=torch.float32)
            collected_next_state = torch.tensor(np.array(collected_next_state), dtype=torch.float32)
            collected_terminated = torch.tensor(np.array(collected_terminated), dtype=torch.float32)
            collected_done = torch.tensor(np.array(collected_done), dtype=torch.float32)

            with torch.no_grad():

                values = critic(collected_state).squeeze(-1)
                next_values = critic(collected_next_state).squeeze(-1)
                advantage = GAE(collected_reward, values, next_values,collected_terminated, collected_done, gamma=gamma, lam=lam)

                target_values = advantage + values

                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)



            for update in range(update_epochs):

                indx = torch.randperm(N)

                for i in range(0, N, batch_size):

                    batch_indx = indx[i:i+batch_size]

                    s = collected_state[batch_indx]
                    a = collected_action[batch_indx]
                    r = collected_reward[batch_indx]
                    n_s = collected_next_state[batch_indx]
                    old_log_p = collected_log_prob[batch_indx]
                    d = collected_done[batch_indx]
                    batch_target_values = target_values[batch_indx]


                    mu, std = actor(s)
                    dist = Normal( mu , std)
                    new_log_prob = dist.log_prob(a).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()   # entropy of the Gaussian policy, summed over 2 action dims, averaged over the batch


                    pred_values = critic(s).squeeze(-1)

                    batch_advantage = advantage[batch_indx]

                    value_loss = ( pred_values - batch_target_values ).pow(2).mean()


                    ratio = torch.exp(new_log_prob - old_log_p)

                    surr1 = ratio * batch_advantage
                    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * batch_advantage
                    policy_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy



                    a_optimizer.zero_grad()
                    policy_loss.backward()
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=max_grad_norm)
                    a_optimizer.step()

                    c_optimizer.zero_grad()
                    value_loss.backward()
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=max_grad_norm)
                    c_optimizer.step()

            D.clear()

        state = next_state
    epoch_rewards_history.append(current_epoch_reward)


torch.save(actor.state_dict(), save_path)

plot_rewards(epoch_rewards_history, epochs, window_size)
