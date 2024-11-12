import torch.nn as nn
import gym
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
import torch
class Q(nn.Module):
    def __init__(self,env):
        super(Q,self).__init__()
        self.Na = env.action_space.n
        try:
            self.N = env.observation_space.n
        except:
            self.N = env.observation_space.shape[0]
        self.linear1 = nn.Linear(self.N +self.Na,16)
        self.linear2 = nn.Linear(16,16)
        self.linear3 = nn.Linear(16,16)
        self.linear4 = nn.Linear(16,16)
        self.linear5 = nn.Linear(16,1)
        self.actv = nn.ReLU()
        self.actions = torch.eye(self.Na)
        self.states = torch.eye(self.N)
    def forward(self,s,a):
        x = torch.concatenate((s,self.actions[a]),dim=0)
        out = self.linear1(x)
        out = self.actv(out)
        out = self.linear2(out)
        out = self.actv(out)
        out = self.linear3(out)
        out = self.actv(out)
        out = self.linear4(out)
        out = self.actv(out)
        out = self.linear5(out)
        return out
class Policy(nn.Module):
    def __init__(self,Q,epsilon = .1):
        super(Policy,self).__init__()
        self.epsilon = epsilon
        self.Q = Q
    def forward(self,s):
        rd = np.random.binomial(1,self.epsilon)
        Qmax = np.argmax([self.Q(torch.Tensor(s),j).item() for j in range(self.Q.Na)])
        out =  rd*torch.randint(0,self.Q.Na,(1,)).item() + Qmax*(1-rd)
        return int(out.item())
def updateQ(Env,state, action,new_state, reward, terminated, Qvalue,Qprim,optimizerQ, Loss,gamma,listLossQ):
    amax = np.argmax([Qvalue(torch.Tensor(new_state),a).item() for a in range(Env.action_space.n)])
    Qprim_maxQvalue = Qprim(torch.Tensor(new_state),amax)
    target = reward + gamma*Qprim_maxQvalue
    target = target.detach()
    optimizerQ.zero_grad()
    Qsa = Qvalue(torch.Tensor(state),action)
    assert Qsa.shape==target.shape, "verifier shape"
    loss = Loss(Qsa.squeeze(),target.squeeze())
    loss.backward()
    optimizerQ.step()
    listLossQ.append(loss.detach().to("cpu"))
    return listLossQ

def swap(Qprim, Qvalue):
    Qprim.load_state_dict(Qvalue.state_dict())
class ChangeReward(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
    def step(self,action):
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        if terminated and reward ==1:
            reward = -2
        else:
            reward = .1
        return new_state, reward, terminated, truncated, _ 

class Renorm(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.mu = None
        self.sigma = None
    def fit(self,N):
        assert type(N)==int
        historic = []
        for i in range(N):
            terminated = False
            truncated = False
            self.reset()
            while(not terminated and not truncated):
                action = self.action_space.sample()
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                state = new_state
                historic.append(state)
        self.close()
        self.mu = np.mean(historic, axis=0)
        self.sigma = np.std(historic,axis=0)
        print(f"statistics over {i} iterations")
        print("mean",self.mu )
        print("std",self.sigma)
    def step(self,action):
        assert self.mu.any() and self.sigma.any(), "call fit() or evaluate mu and mean attributes before calling step()"
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        new_state = (new_state-self.mu)/self.sigma
        return new_state, reward, terminated, truncated, _ 


