import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import os

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha,device_name="cpu",
                 fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                  nn.Linear(*input_dims, fc1_dims),
                  nn.ReLU(),
                  nn.Linear(fc1_dims, fc2_dims),
                  nn.ReLU(),
                  nn.Linear(fc2_dims, 1))
        self.alpha = alpha

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        print("critic device name :", device_name)
        self.device = T.device(device_name) 
        #T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def device(self):
        return next(self.parameters()).device

    def forward(self, state):
        value = self.critic(state)

        return value
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
    
    def get_optimizer_parameters(self):
        return self.optimizer.state_dict()

    def get_critic_parameters(self):
        return self.state_dict()

    def load_parameters(self, optimizer_para, critic_para):
        self.load_state_dict(critic_para)
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        if optimizer_para: 
            self.optimizer.load_state_dict(optimizer_para)
