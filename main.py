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
from utils import Q,Policy, updateQ, swap
import gym
import json
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
    Env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode ="human")
    Qvalue = Q(Env)
    Qprim = Q(Env)
    policy = Policy(Qvalue)
    optimizerQ = optim.Adam(Qvalue.parameters(), lr = lr)
    if start>0:
        Qvalue.load_state_dict(torch.load(os.path.join(loadpath,f"q_load_{start}.pt"), weights_only=True))
        optimizerQ.load_state_dict(torch.load(os.path.join(loadopt,f"opt_q_load_{start}.pt"), weights_only=True))
    Loss = nn.MSELoss()
    listLossQ =[]
    if train:
        for j in tqdm(range(num_episodes)):
            if j>500:
                policy.epsilon = 0.1
            swap(Qprim, Qvalue)
            Env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)#, render_mode ="human")
            truncated = False
            terminated = False
            state =  Env.reset()[0]
            while(not terminated and not truncated):
                action = policy(state)
                new_state, reward, terminated, truncated, _ = Env.step(action)
                if reward ==0 and terminated == True:
                    reward = -1.0/2
                if new_state ==state:
                    reward += -1.0/3
                for k in range(K):
                    updateQ(Env,state,action, new_state, reward, terminated, Qvalue,Qprim, optimizerQ, Loss,gamma, listLossQ)
                state = new_state
            Env.close()
            if (j+1)%100==0:
                torch.save(Qvalue.state_dict(), os.path.join(loadpath,f"q_load_{j+1}.pt"))
                torch.save(optimizerQ.state_dict(), os.path.join(loadopt,f"opt_q_load_{j+1}.pt"))
        plt.plot(listLossQ)
        plt.show()


    Env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode ="human")
    truncated = False
    terminated = False
    state =  Env.reset()[0]
    policy.epsilon = 0
    if test_mode:
        while(not terminated and not truncated):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = Env.step(action)
            if reward ==0 and terminated == True:
                reward = -1
            for k in range(K):
                updateQ(Env,state,action, new_state, reward, terminated, Qvalue,Qprim, optimizerQ, Loss,gamma, listLossQ)
            state = new_state
        Env.close()
