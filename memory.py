import numpy as np
import torch as T
from time import process_time

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.gae = []
        self.next_state = []
        self.tree_idx = -1
        self.sampled =0
        self.batch_size = batch_size
        self.priority_list = []
        self.priority =0
        self.mem_idx = -1

    def set_index(self, idx):
        self.mem_idx = idx

        
    def get_index(self):
        return self.mem_idx

    def clear_gae(self):
        #print("clear gae")
        self.gae = []
    
    def add_gae(self, val):
        self.gae.append(val)

    def add_tree_idx(self, idx):
        #print("tree index:", idx)
        self.tree_idx = idx

    def get_tree_idx(self):
        #print("get tree index:", self.tree_idx)
        return self.tree_idx

    def get_priority_list(self):
        return self.priority_list

    def get_score(self):
        print("sum of rewards :", sum(self.rewards))
        return sum(self.rewards)

    def get_priority(self):
        #print("mean rewards :", sum(self.rewards))
        k = 0.7
        no_replay = 1

        if self.sampled > 0:
           no_replay = self.sampled
    
        p =  (k*max(self.gae) + (1-k) * sum(self.rewards))/no_replay

        if not (p == self.priority):
            self.priority = p
            self.priority_list.append(p)

        return p


    def cal_gae_one_step(self, gamma, device, critic):
        #print("ppomemory states length :", len(self.states))
        times_l = []
        for i in range(len(self.states)):

            t1_start = process_time()
            state = T.tensor(self.states[i], dtype=T.float).to(device)
            next_s = T.tensor(self.next_state[i], dtype=T.float).to(device)
            critic_value = critic(state)
            target_v = critic(next_s)
            val = T.squeeze(critic_value).tolist()
            target_v = T.squeeze(target_v).tolist()

            t1_stop = process_time()
            #print("calc_time:", t1_stop-t1_start)
            times_l.append(t1_stop-t1_start)
            error = (self.rewards[i]+gamma *target_v*(1-int(self.dones[i])) - val)

            #print("add gae")
            self.add_gae(error)

            #agent.update_priority(idx, p)
        return np.sum(times_l)        

    def generate_batches(self):
        n_states = len(self.states)

        #print("n_state:", n_states)
        batch_start = np.arange(0, n_states, self.batch_size)

        indx = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indx)

        batches = [indx[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
               np.array(self.actions),\
               np.array(self.probs),\
               np.array(self.vals),\
               np.array(self.rewards),\
               np.array(self.dones),\
               np.array(self.next_state),\
               batches

    def add_replayTime(self):
        self.sampled+=1

    def get_replayTimes(self):
        #print("number of times sampled :", self.sample)
        return self.sampled

    def store_exp(self, state, action, prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        #self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        #self.gae.append(gae)
        #self.next_state.append(next_state)


    def cal_gae(self, gamma, gae_lambda):
        
        adv = np.zeros(len(self.rewards), dtype=np.float32)

        for t in range(len(self.rewards)-1):
            discount = 1
            a_t =0
            #print("t:", t)
            for k in range(t, len(self.rewards)-1):
                delta = discount * (self.rewards[k]+gamma* self.vals[k+1]*
                (1-int(self.dones[k])) - self.vals[k])
                discount *= (gamma *gae_lambda)
                a_t = a_t + delta

            adv[t] = a_t
        
        return adv    


    def get_gae(self):
        return self.gae

    def store_memory(self, state, action, prob, val, reward, done, gae, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        self.gae.append(gae)
        self.next_state.append(next_state)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.gae = []
        self.next_state = []
        self.sampled = 0


