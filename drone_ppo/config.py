# All hyperparameters 

gamma = 0.99
lam = 0.95                  # GAE lambda
eps = 0.2                   # PPO clip epsilon
learning_rate = 0.0003
ent_coef = 0.01             # entropy bonus coefficient
epochs = 2000               # training episodes (called "epochs" in the original script)
N = 2048                    # rollout size (steps collected before each update)
batch_size = 64
update_epochs = 10          # passes over each rollout
max_grad_norm = 0.5

window_size = 50            # moving-average window for the reward plot
save_path = 'ppo(1000).pth' # change this before a new run so you don't overwrite old weights
