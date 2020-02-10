import gym
import numpy as np
from gym.wrappers import Monitor

from stable_baselines.common.policies import MlpPolicy as common_MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPG_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import PPO1, PPO2, DDPG
#defining the variables
RL_method = "PPO1"
experiment_ID = "handtest_rot_pool_with_MC_C_task4/"
mc_cntr = 10
sensory_value = 0
sesnory_value_str = "sensory_{}".format(sensory_value)
save_name_extension = RL_method
log_dir_read = "./logs/{}/MC_{}/{}/{}/".format(experiment_ID, mc_cntr, RL_method, sesnory_value_str)
log_dir_write = "./logs/{}/videos/MC_{}/{}/{}/".format(experiment_ID, mc_cntr, RL_method, sesnory_value_str)

# defining the environments
env = gym.make('HandManipulate-v1{}'.format(sensory_value))
#env = DummyVecEnv([lambda: env])

# loading the trained model
if RL_method == "PPO1":
    model = PPO1.load(log_dir_read +"/model")
elif RL_method == "PPO2":
    model = PPO2.load(log_dir_read+"/model.pkl")
    env = DummyVecEnv([lambda: env])
elif RL_method == "DDPG":
    model = DDPG.load(log_dir_read+"/model")
    env = DummyVecEnv([lambda: env])
else:
    raise ValueError("Invalid RL mode")
# setting the environment

model.set_env(env)

env_run = gym.make('HandManipulate-v1{}'.format(sensory_value))
#env_run = Monitor(env_run,log_dir_write,force=True)
#model = DDPG.load("PPO2-HalfCheetah_nssu-v3_test2")
obs = env_run.reset()
 #while True:
run_length_seconds = 30
time_step = .01
frame_skip = 5
run_samples = int(np.round(run_length_seconds/(time_step*frame_skip)))
print(run_samples)
rew_sum=0

for ii in range(run_samples):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env_run.step(action)
    if ii == 0:
    	initial_pos = obs[0]
    else:
      	final_pos = obs[0]
    rew_sum=rew_sum+rewards
    env_run.render()

print("avg reward:", rew_sum/run_samples)
print("displacement1:", run_length_seconds*rew_sum/run_samples)
print("displacement2:", final_pos-initial_pos)
print("initial pos:", initial_pos)
print("final pos:", final_pos)

#import pdb; pdb.set_trace()


# python -m baselines.run --alg=ppo1 --env=HandManipulate-v1 --load_path=~/models/pong_20M_ppo2 --play
