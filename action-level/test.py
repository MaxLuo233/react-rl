import gym
env = gym.make('MountainCar-v0')
import numpy as np
np.bool = np.bool_

print(env.observation_space)
print(env.action_space)

# 重置环境，虽说是重置，第一次使用env前，也要调用
# reset会返回初始的observation
observation = env.reset()
print("The initial observation is {}".format(observation))

# 从动作中随机选择一个，random_actio
random_action = env.action_space.sample()
print("The random action is {}".format(random_action))

# 调用step，执行上述的动作，得到下一个状态
new_obs, reward, done, info = env.step(random_action)[:4]
print("The new observation is {}".format(new_obs))
print("The reward is {}".format(reward))
