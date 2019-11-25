from stable_baselines import PPO2_SSup
from stable_baselines import PPO2


for seed in [1, 123, 1245, 1256]:
    # model_ppo2 = PPO2('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./logs-new3/ppo2_CartPole/",
    #                   n_steps=128, seed=seed)
    model_ppo2_ssup_cf03_sp08_sg1 = PPO2_SSup('MlpPolicy', 'CartPole-v1', verbose=1, n_steps=128,
                                               tensorboard_log="./logs-new3/ppo2_ssup_CartPole/",
                                               ssup_coef=0.3, ssup_sample=0.8, ssup_sigma=1., seed=seed)
    model_ppo2_ssup_cf05_sp08_sg1 = PPO2_SSup('MlpPolicy', 'CartPole-v1', verbose=1, n_steps=128,
                                               tensorboard_log="./logs-new3/ppo2_ssup_CartPole/",
                                               ssup_coef=0.5, ssup_sample=0.8, ssup_sigma=1., seed=seed)
    model_ppo2_ssup_cf08_sp08_sg1 = PPO2_SSup('MlpPolicy', 'CartPole-v1', verbose=1, n_steps=128,
                                               tensorboard_log="./logs-new3/ppo2_ssup_CartPole/",
                                               ssup_coef=0.8, ssup_sample=0.8, ssup_sigma=1., seed=seed)
    model_ppo2_ssup_cf12_sp08_sg1 = PPO2_SSup('MlpPolicy', 'CartPole-v1', verbose=1, n_steps=128,
                                               tensorboard_log="./logs-new3/ppo2_ssup_CartPole/",
                                               ssup_coef=1.2, ssup_sample=0.8, ssup_sigma=1., seed=seed)

    # model_ppo2.learn(total_timesteps=500000, tb_log_name="baseline_s%d" % seed)
    model_ppo2_ssup_cf03_sp08_sg1.learn(total_timesteps=500000, tb_log_name="ssup_cf03_sp08_sg1_s%d" % seed)
    model_ppo2_ssup_cf05_sp08_sg1.learn(total_timesteps=500000, tb_log_name="ssup_cf05_sp08_sg1_s%d" % seed)
    model_ppo2_ssup_cf08_sp08_sg1.learn(total_timesteps=500000, tb_log_name="ssup_cf08_sp08_sg1_s%d" % seed)
    model_ppo2_ssup_cf12_sp08_sg1.learn(total_timesteps=500000, tb_log_name="ssup_cf12_sp08_sg1_s%d" % seed)

# import gym
#
# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN
#
# env = gym.make('BeamRider-v0')
#
# model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="../VexGreedy/beamrider/")
# model.learn(total_timesteps=10000000, tb_log_name="baseline")
#
# obs = env.reset()
