import warnings
from torch.nn.utils.rnn import pad_sequence
warnings.filterwarnings("ignore")
from torch import multiprocessing
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torch.distributions import Categorical
import torch.optim as optim
from utils import Q,Policy, updateQ, swap,ChangeReward,Renorm
import gym
import json
import numpy as np


if __name__=="__main__":
    device = torch.device("cpu")
    with open("arg.json","r") as f:
        data = json.load(f)
    train = data["train"]
    test_mode = data["test_mode"]
    start = data["start"]
    epsilon = data["epsilon"]
    gamma = data["gamma"]
    lr = data["lr"]
    num_episodes = data["n_episodes"]
    loadpath = data["loadpath"]
    loadopt = data["loadopt"]
    K = data["K"]
    pathImage = data["pathImage"]
    Env1 = gym.make("CartPole-v1")
    Env1 = Renorm(Env1)
    #Env1.fit(500)
    Env1.mu = np.array([0.00127268,  0.00672972,  0.00110204, -0.00916324])
    Env1.sigma = np.array([0.00127268,  0.00672972,  0.00110204, -0.00916324])
    Qvalue = Q(Env1)
    Qprim = Q(Env1)
    optimizerQ = optim.Adam(Qvalue.parameters(), lr = lr)
    if start>0:
        Qvalue.load_state_dict(torch.load(os.path.join(loadpath,f"q_load_{start}.pt"), weights_only=True))
        optimizerQ.load_state_dict(torch.load(os.path.join(loadopt,f"opt_q_load_{start}.pt"), weights_only=True))
    policy = Policy(Qvalue)
    Loss = nn.MSELoss()
    listLossQ =[]
    listRetour = []
    if train:
        Env = gym.make("CartPole-v1")#, render_mode ="human")
        Env = ChangeReward(Env)
        Env = Renorm(Env)
        Env.sigma = Env1.sigma
        Env.mu = Env1.mu
        for j in tqdm(range(start,start+num_episodes)):
            state =  Env.reset()[0]
            if j==0:
                policy.epsilon = 1
            if j>50:
                policy.epsilon = 0
            swap(Qprim, Qvalue)
            truncated = False
            terminated = False
            retour  = 0
            while(not terminated and not truncated):
                action = policy(state)
                new_state, reward, terminated, truncated, _ = Env.step(action)
                for k in range(K):
                    updateQ(Env,state,action, new_state, reward, terminated, Qvalue,Qprim, optimizerQ, Loss,gamma, listLossQ)
                state = new_state
                retour +=gamma*reward
            Env.close()
            listRetour.append(retour)
            if (j+1)%100==0:
                torch.save(Qvalue.state_dict(), os.path.join(loadpath,f"q_load_{j+1}.pt"))
                torch.save(optimizerQ.state_dict(), os.path.join(loadopt,f"opt_q_load_{j+1}.pt"))
            if (j+1)%20==0:
                plt.figure()
                plt.plot(listLossQ)
                plt.savefig(os.path.join(pathImage,"Loss"))
                plt.figure()
                plt.plot(listRetour)
                plt.savefig(os.path.join(pathImage,"Retour"))

    Env = gym.make("CartPole-v1", render_mode ="human")
    Env = ChangeReward(Env)
    Env = Renorm(Env)
    Env.sigma = Env1.sigma
    Env.mu = Env1.mu
    truncated = False
    terminated = False
    state =  Env.reset()[0]
    policy.epsilon = 0
    
    if test_mode:
        retour = 0
        iterations = 0
        while(not terminated and not truncated):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = Env.step(action)
            retour+=gamma*reward
            iterations += 1
            for k in range(K):
                updateQ(Env,state,action, new_state, reward, terminated, Qvalue,Qprim, optimizerQ, Loss,gamma, listLossQ)
            state = new_state
        Env.close()
        print("retour :", retour)
        print("iterations:", iterations)
