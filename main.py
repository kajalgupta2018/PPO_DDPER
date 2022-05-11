from critic import CriticNetwork
import torch.multiprocessing as mp
from agent import Agent
from learner import Learner
from actor import ActorNetwork
import torch as T
import psutil

def main():
    env_id = 'CartPole-v0'
    N = 20
    batch_size = 5
    n_epochs = 4
    input_dims = [4]
    N_GAMES = 3000
    PRIORITY_ENABLE = False

    T_MAX = 5
    alpha = 0.0003
    n_actions = 2
    if 0:
        optim = SharedAdam(global_actor.parameters(), lr=alpha,
                       betas=(0.92, 0.999))
    else:
        optim = 0

    #mp.set_start_method('spawn')

    #global_actor.share_memory()
    global_ep = mp.Value('i', 0)
    global_steps = mp.Value('i', 0)
    global_shutdown = mp.Value('b', False)
    manager = mp.Manager()
    process_queue = manager.Queue(mp.cpu_count()-1)
    memory_queue = manager.Queue(1500* mp.cpu_count())
    common_dict = manager.dict()
    if 0:
        global_actor = ActorNetwork(n_actions, input_dims, alpha, 0)
        global_actor.share_memory()
        global_critic = CriticNetwork(input_dims, alpha)
        global_critic.share_memory()
    else:
        global_actor = 0
        global_critic = 0

    event_list = [manager.Event() for i in range(mp.cpu_count()-1)]
    agent_list = [Agent(n_actions, input_dims,global_critic,global_actor, 
                        memory_queue, event_list[i],N_GAMES,PRIORITY_ENABLE,
                        env_id, global_ep, global_steps, global_shutdown,optim,
                        process_queue,common_dict,{i+1}, alpha=alpha, 
                        batch_size=batch_size, N=N, n_epochs=n_epochs) 
                        for i in range(mp.cpu_count()-1)]

#    global_learner = Learner(agent_list, memory_queue, event_list) 
#if 0:  
    global_learner = Learner(agent_list, 
                            memory_queue, event_list,global_critic, 
                            global_shutdown, n_actions, global_actor,
                            global_ep,global_steps, T_MAX, input_dims,
                             process_queue,common_dict,{0}, 
                             alpha=alpha, 
                             n_epochs=n_epochs, 
                             batch_size=batch_size, N=N,N_GAMES=N_GAMES)

    #global_learner.cpu_affinity(0)

    #[w.cpu_affinity(i) for i in range(i+1, mp.cpu_count()-1)]

    [w.start() for w in agent_list]
    
    global_learner.start()

    [w.join() for w in agent_list]
    global_shutdown.value = True
    global_learner.join()

if __name__ == "__main__":
    print("cpu core :", mp.cpu_count())
    mp.set_start_method('spawn')
    main()
