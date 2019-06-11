import gym
import random
import math
import numpy as np
env = gym.make('CartPole-v0')

alpha = 0.001
#theta = np.array([ 337.24382213,  -42.56549589, -315.47191556, -526.7562287, 12.65923034])
theta = np.array([random.uniform(-1, 1) for i in range(5)])

def calculatePolicy(observation):
    observation = np.append(observation, [[1]])
    threshold = 1 / (1 + math.exp(-np.dot(theta, observation)))
    val = random.random()
    return (0 if val < threshold else 1, threshold)

def calculateGradient(action, val, observation):
    observation = np.append(observation, [[1]])
    if action == 0:
        return np.array((1 - val) * observation)
    else:
        return np.array(-val * observation)

def run():
    global theta
    observation = env.reset()
    total_reward = 0
    total_gradient = np.zeros(5)

    for t in range(200):
        env.render()
        action, val = calculatePolicy(observation)
        total_gradient += calculateGradient(action, val, observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    theta += alpha * total_gradient * total_reward
    print(theta)
    
for i in range(100000):
    print(i)
    run()
env.close()

