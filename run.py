from stable_baselines import PPO2_SSup
from stable_baselines import PPO2

model_ppo2 = PPO2('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./ppo2_cartpole_long/")
model_ppo2_ssup = PPO2_SSup('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./ppo2_ssup_cartpole_long/")

# baseline = def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
#                  max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
#                  verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
#                  full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):


model_ppo2.learn(total_timesteps=100000,
                 tb_log_name="baseline")
model_ppo2_ssup.learn(total_timesteps=100000,
                      tb_log_name="todo")
