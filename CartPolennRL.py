import gym
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
env = gym.make('CartPole-v0')

#learning rate
alpha = 5e-6


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1, 4, 20, 2

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
print(x.dtype)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

loss_fn = torch.nn.CrossEntropyLoss()

def calculatePolicy(observation):
    observation = torch.from_numpy(observation).float()
    #print(observation)
    
    y_pred = model(observation)
    probabilities = F.softmax(y_pred)
    val = random.random()
    if val < probabilities[0]:
        return 0, y_pred
    return 1, y_pred

def calculateGradient(action, y_pred, observation):
    #We are not calling model.zero_grad() because we want to sum over the gradients of the log of the policies

    loss = loss_fn(y_pred.view(1, 2), torch.tensor([action]))
    loss.backward()

def run(index):
    global theta
    observation = env.reset()
    total_reward = 0

    for t in range(300):
        if (index % 1000 == 0):
            env.render()
        action, y_pred = calculatePolicy(observation)
        calculateGradient(action, y_pred, observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    
    with torch.no_grad():
        for param in model.parameters():
            #cross entropy is -log(P(a_t | s_t)
            param -= alpha * param.grad * total_reward

    #print(total_reward)
    #We can zero out the gradients now (we have computed the sum of the gradients)
    model.zero_grad()   
 
for i in range(30000):
    print(i)
    run(i)

env.close()

