from stable_baselines import PPO2_SSup
from stable_baselines import PPO2

for seed in [1, 123, 1245, 1256]:
    model_ppo2 = PPO2('CnnPolicy', 'SpaceInvaders-v0', verbose=1, tensorboard_log="./logs/ppo2_SpaceInvaders/",
                      n_steps=128, cliprange=0.1, cliprange_vf=-1, seed=seed)
    model_ppo2_ssup_cf05_sp08_sg01 = PPO2_SSup('CnnPolicy', 'SpaceInvaders-v0', verbose=1, n_steps=128,
                                               tensorboard_log="./logs/ppo2_ssup_SpaceInvaders/",
                                               cliprange=0.1, cliprange_vf=-1,
                                               ssup_coef=0.5, ssup_sample=0.8, ssup_sigma=0.1, seed=seed)
    model_ppo2_ssup_cf08_sp08_sg01 = PPO2_SSup('CnnPolicy', 'SpaceInvaders-v0', verbose=1, n_steps=128,
                                               tensorboard_log="./logs/ppo2_ssup_SpaceInvaders/",
                                               cliprange=0.1, cliprange_vf=-1,
                                               ssup_coef=0.8, ssup_sample=0.8, ssup_sigma=0.1, seed=seed)
    model_ppo2_ssup_cf05_sp08_sg001 = PPO2_SSup('CnnPolicy', 'SpaceInvaders-v0', verbose=1, n_steps=128,
                                                tensorboard_log="./logs/ppo2_ssup_SpaceInvaders/",
                                                cliprange=0.1, cliprange_vf=-1,
                                                ssup_coef=0.5, ssup_sample=0.8, ssup_sigma=0.01, seed=seed)
    model_ppo2_ssup_cf08_sp08_sg001 = PPO2_SSup('CnnPolicy', 'SpaceInvaders-v0', verbose=1, n_steps=128,
                                                tensorboard_log="./logs/ppo2_ssup_SpaceInvaders/",
                                                cliprange=0.1, cliprange_vf=-1,
                                                ssup_coef=0.8, ssup_sample=0.8, ssup_sigma=0.01, seed=seed)

    model_ppo2.learn(total_timesteps=1000000, tb_log_name="baseline_s%d" % seed)
    model_ppo2_ssup_cf05_sp08_sg01.learn(total_timesteps=1000000, tb_log_name="ssup_cf05_sp08_sg01_s%d" % seed)
    model_ppo2_ssup_cf08_sp08_sg01.learn(total_timesteps=1000000, tb_log_name="ssup_cf08_sp08_sg01_s%d" % seed)
    model_ppo2_ssup_cf05_sp08_sg001.learn(total_timesteps=1000000, tb_log_name="ssup_cf05_sp08_sg001_s%d" % seed)
    model_ppo2_ssup_cf08_sp08_sg001.learn(total_timesteps=1000000, tb_log_name="ssup_cf08_sp08_sg001_s%d" % seed)
