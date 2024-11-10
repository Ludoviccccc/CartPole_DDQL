import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
import torch
class Q(nn.Module):
    def __init__(self,env):
        super(Q,self).__init__()
        self.Na = env.action_space.n
        self.N = env.observation_space.n
        self.linear1 = nn.Linear(self.N +self.Na,16)
        self.linear2 = nn.Linear(16,16)
        self.linear3 = nn.Linear(16,16)
        self.linear4 = nn.Linear(16,16)
        self.linear5 = nn.Linear(16,1)
        self.actv = nn.ReLU()
        self.actions = torch.eye(self.Na)
        self.states = torch.eye(self.N)
    def forward(self,s,a):
        x = torch.concatenate((self.states[s],self.actions[a]),dim=0)
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
    def __init__(self,Q,epsilon = 1):
        super(Policy,self).__init__()
        self.epsilon = epsilon
        self.Q = Q
    def forward(self,s):
        assert type(s)==int
        rd = np.random.binomial(1,self.epsilon)
        Qmax = np.argmax([self.Q(s,j).item() for j in range(self.Q.Na)])
        out =  rd*torch.randint(0,self.Q.Na,(1,)).item() + Qmax*(1-rd)
        return int(out.item())

def updateQ(Env,state, action,new_state, reward, terminated, Qvalue,Qprim,optimizerQ, Loss,gamma,listLossQ):
    amax = np.argmax([Qvalue(new_state,a).item() for a in range(Env.action_space.n)])
    Qprim_maxQvalue = Qprim(new_state,amax)
    target = reward + gamma*Qprim_maxQvalue
    target = target.detach()
    optimizerQ.zero_grad()
    Qsa = Qvalue(state,action)
    assert Qsa.shape==target.shape, "verifier shape"
    loss = Loss(Qsa.squeeze(),target.squeeze())
    loss.backward()
    optimizerQ.step()
    listLossQ.append(loss.detach().to("cpu"))
    return listLossQ

#def updateQ(Samp,Qvalue,Qprim,optimizerQ, Loss,gamma,listLossQ):
#    T_ = torch.zeros((len(Samp["action"]),Qvalue.Na))
#    for j,a in enumerate(Qvalue(Samp["next"]["observation"]).argmax(dim=1)):   
#        T_[j,a] = 1
#    Qprim_maxQvalue = torch.mul(Qprim(Samp["next"]["observation"]),T_).sum(dim=1)
#    target = Samp["next"]["reward"].squeeze() + gamma*Qprim_maxQvalue
#    target = target.detach()
#    optimizerQ.zero_grad()
#    states = torch.eye(Qvalue.Na)
#    T_action = torch.zeros((len(target),Qvalue.Na))
#    for j,a in enumerate(Samp["action"]):   
#        T_action[j,a] = 1
#    Qsa = torch.mul(Qvalue(Samp["observation"]),T_action).sum(dim=1)
#    assert Qsa.shape==target.shape, "verifier shape"
#    loss = Loss(Qsa.squeeze(),target.squeeze())
#    loss.backward()
#    optimizerQ.step()
#    listLossQ.append(loss.detach().to("cpu"))
#    return listLossQ
def swap(Qprim, Qvalue):
    Qprim.load_state_dict(Qvalue.state_dict())
