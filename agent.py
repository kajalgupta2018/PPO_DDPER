import torch.multiprocessing as mp
import gym
from torch.distributions import Categorical
import torch as T
import numpy as np
import random
import threading
from SumTree import SumTree
from memory import PPOMemory
from actor import ActorNetwork
from critic import CriticNetwork
import os
import time
from time import process_time

from utils import Logger

class Agent(mp.Process):

    def __init__(self, n_actions, input_dims, critic, actor, memory_queue,
                 c_event, N_GAMES,pri_enable,
                 env_id, global_ep_idx, global_l_steps , g_shutdown, optimizer,
                 process_queue, common_dict,mask,
                 gamma=0.99, alpha=0.0003, gae_lambda=0.95, 
                 policy_clip=0.2, batch_size=64, 
                 N=2048, n_epochs=10):

        super(Agent, self).__init__()
        self.gamma = gamma
        #self.optimizer = optim
        self.n_actions = n_actions
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.global_l_steps = global_l_steps
        self.N = N
        self.N_GAMES = N_GAMES
        self.n_epochs = n_epochs
        self.device_name = 'cpu'
        if actor ==0:
            self.g_actor = True
            self.actor = ActorNetwork(n_actions, input_dims, alpha, 0)
            self.critic = CriticNetwork(input_dims, alpha)
        else:
            self.actor = actor
            self.g_actor = True
            self.critic = critic #CriticNetwork(input_dims, alpha)
        self.process_queue = process_queue
        self.common_dict = common_dict
        self.memory_list = []
        #self.actor.share_memory()
        self.tree = SumTree(2000)
        self.memory = PPOMemory(batch_size)
        self.critic_q = memory_queue
        self.env = gym.make(env_id) 
        self.episode_idx = global_ep_idx
        self.c_event = c_event
        self.g_shutdown = g_shutdown
        self.pri_enable = pri_enable
        self.a_pid = 0
        self.prb = 0
        self.mask = mask
        self.logs = 0
        self.log_mem =0
        self.avg_pri= []
        #self.log_mem.append(df)
        self.n_obj =[]
        self.learn_steps =[]
        self.n_samples =[]
        self.avg_gae =[]
        self.no_episode =[]
        self.col_name = ["episode_no","learned_steps",
                 "no_obj",
                  "n_samples",
                 "avg_priority",
                 "avg_gae"]

          
        #print("agent mask :", mask)
        #self.env_queue =   

        #self.Priority_Queue(self.tree, 
        #                            self.critic_q, self.c_event)

    def run(self):

        n_steps = 0
        if T.cuda.is_available():
            os.sched_setaffinity(self.pid,self.mask)

        self.prb = self.Priority_Queue(self.tree, 
                                   self.critic_q, self.c_event, 
                                   self.memory_list, self.pri_enable, self.pid)
        self.a_pid = self.pid
        #print("agent : pid :", self.a_pid)
        #print("agent device :", self.actor.device)


        self.log_mem = Logger([], "agent_mem_"+str(self.a_pid)) 

        self.logs = Logger(['score','episode_no'], 
                           "agent_"+str(self.a_pid))

        #self.logs.print_msg("agent :pid", self.a_pid)
        self.prb.start()
        self.load_parameters()
        learn_iters = 0
        avg_score = 0
        chk_priority = False 
        score_history = []
        episode_idx = []
        best_score_l = []
        avg_score_l = []
        best_score = self.env.reward_range[0]
        learner_steps = []
        n_steps_l = []

        while not self.g_shutdown.value:
            #print("i am agent")
            if self.episode_idx.value < self.N_GAMES:
                done = False
                #print("episode :" ,self.episode_idx.value)
                observation = self.env.reset()
                score =0
                #print("done :", done)
                while not done:
                    
                    action, prob , val = self.choose_action(observation)

                    observation_, reward, done, info = self.env.step(action)
                    
                    n_steps += 1
                    #print("n_steps :", n_steps)
                    score += reward
                    
                    self.remember(observation, action, 
                                  prob, val, reward, done, observation_)

                    if n_steps % 200 == 0:
                       chk_priority = not chk_priority
                       #self.update_priority_batch()
                       #print("loading parameters",self.episode_idx.value )

                    if n_steps % self.N == 0:
                        #print("n_steps :", n_steps)
                        self.add_priority()
                        learn_iters +=1
                        with self.episode_idx.get_lock():
                            self.episode_idx.value += 1


                    observation = observation_


                self.load_parameters()
                score_history.append(score)
                episode_idx.append(self.episode_idx.value)
                learner_steps.append(self.global_l_steps.value)
                avg_score = np.mean(score_history[-100:])
                #print("episode :",i, "score:%.1f"% score)
                if avg_score > best_score:
                    best_score = avg_score
                
                avg_score_l.append(avg_score)
                best_score_l.append(best_score)
                n_steps_l.append(n_steps)
                    #print('episode', self.episode_idx.value, 'score %.1f' % score)

                if len(episode_idx) % 10 == 0:

                    #self.print_stats()
                    print('episode:', self.episode_idx.value, 'score %.1f' % score, 'learner_steps',
                     self.global_l_steps.value, "best_score %.1f"%best_score,
                     "avg_score %.1f"%avg_score)

                #print('episode', self.episode_idx.value, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                 #   'time_steps', n_steps, 'learning_steps', learn_iters)
            else:
                break
        self.print_stats()
        self.logs.add("score",score_history)
        self.logs.add("learned_steps",learner_steps)
        self.logs.add("episode_no",episode_idx)
        self.logs.add("avg_score",avg_score_l)
        self.logs.add("best_score",best_score_l)
        self.logs.add("no_steps",n_steps_l)
        if self.pri_enable:
            self.log_mem.add(self.col_name[0], self.no_episode)
            self.log_mem.add(self.col_name[1], self.learn_steps)
            self.log_mem.add(self.col_name[2], self.n_obj)
            self.log_mem.add(self.col_name[3], self.n_samples)
            self.log_mem.add(self.col_name[4], self.avg_pri)
            self.log_mem.add(self.col_name[5], self.avg_gae)
            self.log_mem.write_to()


        #self.logs.append({"score":'100'})
        self.logs.write_to()
        self.prb.stop() 
        self.prb.join()
        self.load_parameters(4)
        print("exiting from agent")

    def add_priority(self):
       
        p = self.memory.get_priority()
        idx = self.tree.add(p, self.memory)
        #print("add tree:", idx)
        #print("add a_pid :", self.a_pid, "tree_idx", idx, "score:", self.memory.get_score())
        self.memory.add_tree_idx(idx)

        self.memory.set_index(len(self.memory_list))
        self.memory_list.append(self.memory)
        self.memory = PPOMemory(self.batch_size)

    def update_priority(self, idx, p):
        self.tree.update(idx, p)

    def load_parameters(self, event_ind=0):
        if event_ind ==4:
            self.process_queue.put((self.a_pid, 0, 4))
            while self.a_pid not in self.common_dict:
               time.sleep(0.0001)
            return   
        #t1_start = process_time() 
              
        self.process_queue.put((self.a_pid, 0, 3))
        while self.a_pid not in self.common_dict:
           time.sleep(0.0001)


        if 1:
            #print("getting parameters :", actor_para)
            (actor_para, critic_para, pri_list) = self.common_dict[self.a_pid]  
            actor_o =0 
            critic_o =0
            #(actor_para, actor_o, critic_para, critic_o) = self.common_dict[self.a_pid]  
            #print("getting parameters :", actor_para)

        #(actor_para, actor_o, critic_para, critic_o) = self.common_dict[self.a_pid]  
        #print("got actor_para", self.a_pid)
        else: 
            (actor_para, critic_para) = self.common_dict[self.a_pid] 
            actor_o = actor_para.get_optimizer_parameters()
            actor_para = actor_para.get_actor_parameters()
            critic_o = critic_para.get_optimizer_parameters()
            critic_para = critic_para.get_critic_parameters()
        
    
        if 1:
            self.actor.load_parameters(actor_o,
                                   actor_para)
            self.critic.load_parameters(
                critic_o,
                critic_para)

        del self.common_dict[self.a_pid]


        #print("pid :", self.a_pid, "pri_list:", len(pri_list))
        for obj in range(len(pri_list)):
            (idx, p, hmemory) = pri_list[obj]
            #print("idx:", idx)
            #print("priority:", p)
            self.update_priority(idx, p)
            i = hmemory.get_index()
            self.memory_list[i] = hmemory
            self.log_mem_stats(i, hmemory)
        #print("pid:", self.a_pid, "process time :", t1_stop-t1_start)
        #self.update_priority_batch()


        

    def log_mem_stats(self, index, mem):
        samples = mem.get_replayTimes()
        p_list = mem.get_priority_list()
        self.avg_pri.append(np.mean(p_list))
        #self.log_mem.append(df)
        self.n_obj.append(index)
        self.learn_steps.append(self.global_l_steps.value)
        self.n_samples.append(samples)
        self.avg_gae.append(np.mean(mem.get_gae()))
        self.no_episode.append(self.episode_idx.value)



    def print_stats(self):
        mdata = self.tree.get_data()
        #print("mdata:", mdata)
        no_obj = 0
        no_sampled = 0
        not_sampled =0 
        avg_pri = []
        n_samples = []
        avg_gae = []
        no_episode = []
        for i in range(len(mdata)):
            mem = mdata[i]
            if(mem !=0):
                no_obj += 1
                samples = mem.get_replayTimes()
                if samples ==0:
                    not_sampled += 1
                else:
                    no_sampled += 1

        print("number of object:", no_obj)    

        print("not sampled:", not_sampled)    
        print("sampled:", no_sampled)    
        #return (no_obj, not_sampled, no_sampled, n_samples, avg_gae, avg_pri)
        

    def remember(self, state, action, probs, vals, reward, done, next_state):
        #self.memory.store_memory(state, action, probs, vals, reward, done)
        if self.g_actor:
            a = np.stack(next_state, axis=0)
            next_s = T.tensor(a, dtype=T.float).to(self.actor.device)
            value = self.critic(next_s)
            value = T.squeeze(value).item()
        else:    
            self.process_queue.put((self.a_pid, next_state, 1))
            while self.a_pid not in self.common_dict:
                time.sleep(0.0001)

            (value) = self.common_dict[self.a_pid]    
            del self.common_dict[self.a_pid]

        error = (reward+self.gamma *value*
                 (1-int(done)) - vals)
        self.memory.store_memory(state, action, probs, vals, reward, done, error, next_state)
        #self.memory.store_exp(state, action, probs, reward, done, next_state)

    def save_models(self):
        #self.actor.save_checkpoint()
        #self.critic.save_checkpoint()
        print("NOT")

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()    

    def get_actor(self):
        return self.actor

    def choose_action(self, observation):
        if self.g_actor:
            #print("choose action", observation)
            a = np.stack(observation, axis=0)
            #print("state :", a)
            state = T.tensor(a, dtype=T.float).to(self.actor.device)
            #print("to tensor", state)
            #state = s.to(self.actor.device)
            #print("device", state)
            dist = self.actor(state)
            value = self.critic(state)
            action = dist.sample()
            probs = T.squeeze(dist.log_prob(action)).item()
            action = T.squeeze(action).item()
            value = T.squeeze(value).item()
        else:
            self.process_queue.put((self.a_pid, observation, 0))
            while self.a_pid not in self.common_dict:
                time.sleep(0.0001)

            (action, probs, value) = self.common_dict[self.a_pid]    
            del self.common_dict[self.a_pid]
        #action = dist.sample()
        #probs = T.squeeze(dist.log_prob(action)).item()
        #action = T.squeeze(action).item()
        #value = T.squeeze(value).item()

        return action, probs, value

    class Priority_Queue(threading.Thread):
        stop_signal = False
        
        def __init__(self, p_tree, c_queue, c_event, memory_list, 
        pri_enable, a_pid):
            threading.Thread.__init__(self)
            self.prb_tree = p_tree
            self.c_queue = c_queue
            self.c_event = c_event
            self.a_pid = a_pid
            self.memory_list = memory_list
            self.empty_queue = False
            self.pri_enable = pri_enable
        
        def stop(self):
            self.stop_signal = True

        def empty_queue(self):
            self.empty_queue = True


        def run(self):

            while not self.stop_signal:
                if self.c_event.is_set():
                    if self.pri_enable:
                        n = 5
                        prb_total = self.prb_tree.total()
                        #print("prb_total:", prb_total)
                        segment = prb_total/n
                        #print("segment", segment)
                        if prb_total > 0:
                            if segment > 0:
                                for i in range(n):
                                    a = segment*i
                                    b = segment * (i+1)
                                    s = random.uniform(a, b)
                                    (idx, priority, hmemory) = self.prb_tree.get(s)
                                    #print("priority:", priority, "idx:", idx)
                                    hmemory.add_replayTime()
                                    self.c_queue.put([self.a_pid, idx, priority, hmemory])

                            else:
                                s = random.uniform(0, prb_total)
                                (idx, priority, hmemory) = self.prb_tree.get(s)
                                hmemory.add_replayTime()
                                #print("index :", idx)
                                self.c_queue.put([self.a_pid, idx, priority, hmemory])

                        self.c_event.clear()
                    else:
                        #print("no priority", len(self.memory_list))
                        while len(self.memory_list)>0:
                            hmemory = self.memory_list[0]
                            idx = -1
                            priority = -1
                            self.c_queue.put([self.a_pid, idx, priority, hmemory])
                            self.memory_list.remove(hmemory)
                    self.c_event.clear()
                else:

                    self.c_event.wait(2)

            print("exiting from thread")
