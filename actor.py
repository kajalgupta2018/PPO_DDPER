import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import os

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, optimizer, 
    device_name="cpu", fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
                  nn.Linear(*input_dims, fc1_dims),
                  nn.ReLU(),
                  nn.Linear(fc1_dims, fc2_dims),
                  nn.ReLU(),
                  nn.Linear(fc2_dims, n_actions),
                  nn.Softmax(dim=-1))
        if optimizer == 0:      
            #print("start new optimizer")
            self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        else:
            #print("old optimizer")
            self.optimizer = optimizer
        print("actor device name :", device_name)    
        self.device = T.device(device_name) 
        #T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.alpha = alpha


    def device(self):
        return next(self.parameters()).device

    def forward(self, state):
        #print("forward", state)
        dist = self.actor(state)
        #print("after_call")
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def get_optimizer_parameters(self):
        return self.optimizer.state_dict()

    def get_actor_parameters(self):
        return self.state_dict()

    def load_parameters(self, optimizer_para, actor_para):
        self.load_state_dict(actor_para)
        #self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        if optimizer_para: 
            self.optimizer.load_state_dict(optimizer_para)

