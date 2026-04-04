import marioai
from multiprocessing import Pool, Manager, current_process
from itertools import cycle
from agents import MLPAgent, CodeAgent
from tasks import MoveForwardTask, HunterTask
import numpy as np
import time

# Variable that configures the number of parallel processes
N_PROCESSES = 5
# Task Definition
TASK_TO_SOLVE = MoveForwardTask#HunterTask




port_list = [4242 + i for i in range(N_PROCESSES)]
def evaluate_agent(agent, task, episodes=1):
    """
    Evaluates the agent on the task for a given number of episodes.
    Returns the average fitness (reward).
    """
    exp = marioai.Experiment(task, agent)
    # Speed up simulation for training
    exp.max_fps = -1 
    
    total_reward = 0

    for _ in range(episodes):
        episode_reward = 0
        task.level_difficulty = 0
        # Try up to 3 levels of increasing difficulty
        for _ in range(3):
            rewards = exp.doEpisodes(1)
            episode_reward += task.cum_reward
            
            if task.status == 1: # WIN
                task.level_difficulty += 1
            else:
                break
        
                
        total_reward += episode_reward
        
    
    return total_reward / episodes


# --- GLOBAL VARIABLES FOR WORKER PROCESSES ---
# These exist independently inside EACH worker process.
worker_task = None 
worker_agent = None

def init_worker(agent_class):
    """
    This runs ONCE when each worker process starts.
    """
    global worker_agent, worker_task
    
    # Each worker needs to pick a port. Since we have 10 workers 
    # and 10 ports, we can use a trick to assign them.
    import multiprocessing
    # Get the index of the current worker (0 through 9)
    # Note: This is a hacky way to get a unique index; 
    # alternatively, use a shared Counter/Queue.
    worker_idx = int(multiprocessing.current_process().name.split('-')[-1]) - 1
    port = port_list[worker_idx % len(port_list)]
    
    #print(f"Worker initialized: Connecting once to port {port}...")

    worker_agent = agent_class()
    if worker_task is None:
        worker_task = TASK_TO_SOLVE(visualization=False, port=port, init_mario_mode=0)


def evaluate_individual(ind_info):
    """
    This runs for every individual in the population.
    It uses the GLOBALLY cached worker_task.
    """
    global worker_task, worker_agent
    
    # 1. Update the persistent agent with the new DNA
    if isinstance(worker_agent, MLPAgent):
        worker_agent.set_param_vector(ind_info)
    elif isinstance(worker_agent, CodeAgent):
        worker_agent.action_function = ind_info

    
    # 2. Run evaluation using the EXISTING connection
    # No "with", no "connect", just use the persistent object.
    try:
        reward = evaluate_agent(worker_agent, worker_task)
    except Exception as e:
        print(f"Error in worker: {e}")
        reward = 0
        
    return reward

def evaluate(agent_class, ind_info):
    global worker_agent, worker_task
    if worker_agent is None:
        worker_agent = agent_class()
    if worker_task is None:
        worker_task = TASK_TO_SOLVE(visualization = False, port=port_list[0])
    return evaluate_individual(ind_info)


worker_pool = None

def get_pool(agent, population_size):
    global worker_pool
    if worker_pool is None:
        n_processes = min(N_PROCESSES, max(1, population_size))
        worker_pool = Pool(processes=n_processes, initializer=init_worker, initargs=(agent,))
    return worker_pool

def evaluate_population(agent, population):
    pool = get_pool(agent, len(population))
    rewards_list = pool.map(evaluate_individual, population)
    
    # We do NOT close the pool here anymore so workers persist across iterations.
    return np.array(rewards_list)