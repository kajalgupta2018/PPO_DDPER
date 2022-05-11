import torch.multiprocessing as mp
import time
import numpy as np
import threading
import torch as T
from actor import ActorNetwork
from critic import CriticNetwork
import torch.optim as optim
import time
import os
import copy
from utils import Logger
#import queue from Queue

class Learner(mp.Process):
    def __init__(self, agent_list, queue, event_list, critic, 
                    g_shutdown, n_actions, actor, g_episode_idx,
                    g_learner_steps,
                    T_MAX, input_dims, process_queue, common_dict,mask,
                    gamma=0.99, alpha=0.0003, 
                    gae_lambda=0.95, policy_clip=0.2, n_epochs=10,
                    batch_size=64, N=128, N_GAMES=300): 
     
        super(Learner, self).__init__()
        self.gamma = gamma
        self.n_actions = n_actions
        self.alpha = alpha
        self.g_learner_steps = g_learner_steps
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.N = N
        self.mask = mask
        self.device_name = 'cuda:0' if T.cuda.is_available() else 'cpu'
        self.event = mp.Event()

        if 1:
            if actor ==0:
                self.actor = ActorNetwork(n_actions, input_dims, 
                alpha,0, self.device_name)
                self.critic = CriticNetwork(input_dims, alpha, self.device_name)
                #self.actor.share_memory()
                #self.critic.share_memory()

            else: 
                self.actor = actor
                self.critic = critic
            #self.optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
            #self.optimizer = SharedAdam(self.actor.parameters(), lr=alpha,
            #                   betas=(0.92, 0.999))

        self.N_GAMES = N_GAMES
        self.n_epochs = n_epochs
        if 0:
            self.agent_list = agent_list
            #self.actor  = actor
        self.T_MAX = T_MAX
        self.c_queue = queue
        self.event_list =  event_list
        self.n_epochs = n_epochs
        self.g_episode_idx = g_episode_idx
        self.g_shutdown = g_shutdown
        self.process_queue = process_queue
        self.common_dict = common_dict
        self.p_queue = 0
        self.l_queue = []
        self.pri_dict = {} 



    def learn(self, a_pid, idx, priority, hmemory, event):

        #actor = agent.get_actor()
        #print(" I am in learner process")
        state_arr = []
        done_arr = []
        next_state_arr = []
        reward_arr = []

        c_loss_l = []
        a_loss_l = []

        for _ in range(self.n_epochs):

            #print(" current epochs :", _)
            state_arr, action_arr, old_probs_arr, vals_arr,\
            reward_arr, done_arr, next_state_arr, batches = \
                    hmemory.generate_batches()

            values = vals_arr
            if 0:
                adv = np.zeros(len(reward_arr), dtype=np.float32)

                for t in range(len(reward_arr)-1):
                    discount = 1
                    a_t =0 
                    #print("t:", t)
                    for k in range(t, len(reward_arr)-1):
                        delta = discount * (reward_arr[k]+self.gamma* values[k+1]*
                        (1-int(done_arr[k])) - values[k])
                        discount *= (self.gamma * self.gae_lambda)
                        a_t = a_t + delta

                    adv[t] = a_t
                    #print("adv[t] :", adv[t])
            adv = hmemory.cal_gae(self.gamma, self.gae_lambda)        
            adv = T.tensor(adv).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                #print("batch :", batch)
                #print("adv_batch :", adv[batch])
                #print("learner device  :", self.actor.device)
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                #print("states:", states)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)

                #print("old_ratio:", old_probs)

                prob_ratio = new_probs.exp() / old_probs.exp()
                #print("prob_ratio:", prob_ratio.data)

                weighted_probs = adv[batch] * prob_ratio

                weighted_clipped_probs = T.clamp(prob_ratio, 
                                                 1-self.policy_clip,
                                                 1+self.policy_clip)*adv[batch]


                #print("weighted_probs :", weighted_probs)    
                #print("weighted_clipped_probs :", weighted_clipped_probs)    
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = adv[batch] + values[batch]



                critic_loss = (returns-critic_value)**2

                critic_loss = critic_loss*prob_ratio.data
                critic_loss = critic_loss.mean()
                #print("critic_loss:", critic_loss)
                #print("actor_loss:", actor_loss)

                total_loss = actor_loss + 0.5*critic_loss


                #print("total_loss:", total_loss)    
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()



                #if self.g_episode_idx.value % self.T_MAX == 0:
                   #print("loading parameters") 
                 #  actor.load_state_dict(
                 #      self.actor.state_dict())


                self.critic.optimizer.zero_grad()
                
                #total_loss.backward()
                critic_loss.backward()
                self.critic.optimizer.step()
                a_loss_l.append(T.squeeze(actor_loss.data).tolist())
                c_loss_l.append(T.squeeze(critic_loss.data).tolist())

        if priority > 0 :
            hmemory.clear_gae()
            for i in range(len(state_arr)):
                states = T.tensor(state_arr[i], dtype=T.float).to(self.actor.device)
                next_s = T.tensor(next_state_arr[i], dtype=T.float).to(self.actor.device)
                critic_value = self.critic(states)
                target_v = self.critic(next_s)
                val = T.squeeze(critic_value).tolist()
                target_v = T.squeeze(target_v).tolist()

                #print("target_v:", target_v)
                error = (reward_arr[i]+self.gamma *target_v*(1-int(done_arr[i])) - val)

                hmemory.add_gae(error)
                hmemory.add_tree_idx(idx)
            p = hmemory.get_priority()
            #print("updated p:", p, "old_priority:", priority)
            if not (a_pid in self.pri_dict):
                #print("keys does not exixts:", a_pid)
                self.pri_dict[a_pid] = []
                #print("keys", self.pri_dict)

            self.pri_dict[a_pid].append((idx , p, hmemory))
            #agent.update_priority(idx, p)

        return (np.mean(a_loss_l), np.mean(c_loss_l))


    def run(self):
        print(" i am in learner")
        logs = Logger([], "learner")

        learned_steps_l = []
        queue_l = []
        if T.cuda.is_available():
            os.sched_setaffinity(self.pid, self.mask)

        if 1:        
            self.p_queue = self.process_thread(self.process_queue,self.event, 
                                       self.common_dict, 
                                       self.actor, 
                                       self.critic, 
                                       self.c_queue, 
                                       self.l_queue, 
                                       self.event_list,
                                       self.pri_dict)

            self.p_queue.daemon = True 
            self.p_queue.start()
            #print("agent list :", len(self.agent_list))
            learn_steps = 0
            a_loss_l = []
            c_loss_l = []

            while not self.g_shutdown.value: 
                           #print(" agent", i)
                   #print("event set", i)
                   
                if len(self.l_queue):            
                    [a_pid, idx, p, memory,event] = self.l_queue.pop(0)
                    #print("priority :", p)
                    self.g_learner_steps.value +=1
                    (actor_loss, critic_loss) = self.learn(a_pid, idx, p, memory, event)
                    learned_steps_l.append(self.g_learner_steps.value)
                    queue_l.append(len(self.l_queue))
                    a_loss_l.append(actor_loss)
                    c_loss_l.append(critic_loss)

                    #self.g_learner_steps +=1
                    #print("no of learned steps:",  self.g_learner_steps.value, len(self.l_queue))
                    #print("actor loss:", actor_loss)
                else:
                    if self.g_episode_idx.value >= self.N_GAMES:
                        self.g_shutdown.value = True
                        time.sleep(2)  
                    else:    
                        self.event.set()
                        time.sleep(.002)  

    
            logs.add("learned_steps", learned_steps_l)
            logs.add("memory_queue_length", queue_l)
            logs.add("actor_loss",a_loss_l)
            logs.add("critic_loss",c_loss_l)

            logs.write_to()
            #self.p_queue.stop()
            self.p_queue.join()
            

    class process_thread(threading.Thread):
        stop_signal = False
        def __init__(self, process_queue, event, common_pid, actor, critic, 
                     c_queue, l_queue, event_list, pri_dict):
            threading.Thread.__init__(self)
            self.process_queue = process_queue
            self.event = event
            self.common_pid = common_pid
            self.actor = actor
            self.critic = critic
            self.c_queue = c_queue
            self.l_queue = l_queue
            self.event_list = event_list
            self.pri_dict = pri_dict
            #print(" i am in process thread")
        
        def stop(self):
            self.stop_signal = True

        def choose_value(self, observation):
            a = np.stack(observation, axis=0)
            state = T.tensor(a, dtype=T.float).to(self.actor.device)
            value = self.critic(state)
            value = T.squeeze(value).item()
            return value

        def choose_action(self, observation):
            a = np.stack(observation, axis=0)
            state = T.tensor(a, dtype=T.float).to(self.actor.device)
            dist = self.actor(state)
            value = self.critic(state)
            action = dist.sample()
            probs = T.squeeze(dist.log_prob(action)).item()
            action = T.squeeze(action).item()
            value = T.squeeze(value).item()

            return action, probs, value

        def run(self):
            #print(" learner process thread")
            end_ind =0
            while not self.stop_signal:
                #print(" while process thread")
                while self.process_queue.empty() == False:
                   #print("process_queue no empty") 
                   [pid, obs, event] = self.process_queue.get_nowait()
                   #print("pid, ", pid, "event", event)
                   if event ==0: 
                       self.common_pid[pid] = self.choose_action(obs)
                   elif event==1: 
                       self.common_pid[pid] = self.choose_value(obs)
                   elif event == 4:
                        end_ind = end_ind +1
                        self.common_pid[pid] = (1)
                        if len(self.event_list) == end_ind:
                           self.stop_signal = True 
                           break 
                   elif event == 3:
                      #print("send parameters", pid)
                      cpu = T.device("cpu")
                      #a = self.actor.to(cpu)
                      #c = self.critic.to(cpu)
                      a = copy.deepcopy(self.actor)
                      c = copy.deepcopy(self.critic)
                      a = a.to(cpu)
                      c = c.to(cpu)
 
                      #self.actor.to(self.actor.device)
                      #self.critic.to(self.critic.device)
                      #a = self.actor.to(cpu)
                      #c = self.critic.to(cpu)
                      actor_para= a.get_actor_parameters()
                      critic_para= c.get_critic_parameters()
                      #actor_opt_para= a.get_optimizer_parameters()
                      #critic_opt_para= c.get_optimizer_parameters()
                      #print("l actor device :", self.actor.device)
                      #print("a device :", a.device)
                      #print("a actor device :", a.device)
                      #print (" is a is actor ", a is self.actor)
                      #self.common_pid[pid] = (a,c)
                      if pid in self.pri_dict.keys():
                         p_list = self.pri_dict[pid]
                         self.pri_dict.pop(pid)
                      else:
                         p_list = []
                      self.common_pid[pid] = (actor_para.copy(), 
                                              critic_para.copy(), 
                                              p_list)
                      #self.common_pid[pid] = (actor_para.copy(), actor_opt_para.copy(), critic_para.copy(),
                      #                        critic_opt_para.copy())
                      

                
                for i in range(len(self.event_list)):
                   #print(" agent", i)
                   event = self.event_list[i]
                   event.set()
                   #print("event set", i)
                   while self.c_queue.empty() == False:
                       [a_pid, idx, p, memory] = self.c_queue.get_nowait()
                       #print("priority :", p)
                       self.l_queue.append((a_pid, idx, p, memory, event))
                       del [a_pid, idx, p, memory]

              
                self.event.wait(2)      
                self.event.clear()
            print("exiting from learner thread")
            

            
