import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):

    def __init__(self, s_size, a_size, seed):
        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(s_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, a_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):

    def __init__(self, s_size, a_size, seed):
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(s_size, 512)
        self.fc2 = nn.Linear(512+a_size, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = F.leaky_relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)