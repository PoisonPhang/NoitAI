import gym
import gym_noita

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env('noita-v0')

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save('a2c_noita_test')

del model

model = A2C.load("a2c_noita_test")

obs = env.reset()

done = False

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()