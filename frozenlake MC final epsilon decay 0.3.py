import gym
import numpy as np
import pandas as pd

env = gym.make("FrozenLake-v1") # 얼음위 미끄러짐 존재

Q = np.zeros([env.observation_space.n, env.action_space.n])
N = np.zeros([env.observation_space.n, env.action_space.n])
R = []
L = [] #for file saving

num_episodes = 1000000
epsilon = 0.1 # On-Policy exploration rate

for i in range(num_episodes):
    state = env.reset()
    reward_sum = 0
    done = False
    trajectory = []
    result_sum = 0.0
    eps = epsilon * (1 - i/num_episodes)
    while not done:
        if np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        new_state, reward, done, _ = env.step(action)
        trajectory.append((state, action))
        state = new_state
        reward_sum += reward
    R.append(reward_sum)

    for (state, action) in trajectory:
        N[state, action] += 1.0
        alpha = 1.0 / N[state, action]
        Q[state, action] += alpha * (reward_sum - Q[state, action]) # Monte Carlo model update

    if i % 1000 == 0 and i != 0:
        L.append(sum(R))
        print(i)

print("Total Episodes : %d, Total Reward : %d"%(i,sum(R)))
df = pd.DataFrame(L)
df.to_csv('eps decay %f, %f, slip try3'%(i, epsilon), index=False)

env.close()