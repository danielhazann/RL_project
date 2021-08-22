from Model import Net

import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)                                     
BATCH_SIZE = 64        
GAMMA = 0.99          
TAU = 0.001               
LR = 0.003
UPD_EVERY = 7

class Agent:

    def __init__(self, s_size, a_size, seed):

        self.s_size = s_size 
        self.a_size = a_size
        self.seed = seed

        self.qnetwork_local = Net(self.s_size, self.a_size, self.seed).to(device)
        self.qnetwork_target = Net(self.n_states, self.n_actions ,self.seed ).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.seed)

        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPD_EVERY

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn

            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()

        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.n_actions))

    def learn(self, experiences, gamma):
        
        for e in experiences:
            state = torch.from_numpy(e.state.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
            action = torch.tensor(e.action).long().to(device)
            reward = torch.tensor(e.reward).float().to(device)
            next_state = torch.from_numpy(e.next_state.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
            done = torch.tensor(e.done).float().to(device)

            # Get max predicted Q values (for next states) from target model
            Q_target_next = self.qnetwork_target(next_state).detach().max(0)[0].unsqueeze(0)
            # Compute Q targets for current states 
            Q_target = reward + (self.gamma * Q_target_next * (1 - done))
            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(state).gather(0, action)

            loss = F.mse_loss(Q_expected, Q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # ------------------- update target network ----------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, TAU):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        return len(self.memory)

class Experience:

    def __init__(self, state, action, reward, next_state, done):

        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
