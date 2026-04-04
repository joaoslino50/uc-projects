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
TASK_TO_SOLVE = HunterTask#MoveForwardTask
# Starting difficulty level (overridden at runtime via evaluate_population)
STARTING_DIFFICULTY = 0
# Fitness penalty applied when Mario dies or times out (not a WIN).
# Large enough to clearly distinguish a surviving run from a dying one.
DEATH_PENALTY = 500




port_list = [4242 + i for i in range(N_PROCESSES)]
def evaluate_agent(agent, task, episodes=1):
    """
    Evaluates the agent on the task for a given number of episodes.
    Returns the average fitness (reward).

    Each sub-episode uses a freshly randomised level seed so the agent
    must generalise across many layouts instead of memorising one fixed map.
    Dying or timing out (status != WIN) incurs a DEATH_PENALTY so the EA
    has a clear gradient toward survival.
    """
    exp = marioai.Experiment(task, agent)
    # Speed up simulation for training
    exp.max_fps = -1

    total_reward = 0

    for _ in range(episodes):
        episode_reward = 0
        task.level_difficulty = STARTING_DIFFICULTY

        # Try up to 3 levels of increasing difficulty
        for _ in range(3):
            # ---- Randomise level layout each sub-episode ----
            # This forces the network to use its vision inputs to generalise
            # instead of hard-coding a single memorised path.
            task.env.level_seed = int(np.random.randint(1, 9999))

            rewards = exp.doEpisodes(1)
            episode_reward += task.cum_reward

            if task.status == 1:   # WIN → try a harder level
                task.level_difficulty += 1
            else:
                # Mario died or ran out of time — penalise this outcome
                episode_reward -= DEATH_PENALTY
                break

        total_reward += episode_reward

    return total_reward / episodes


# --- GLOBAL VARIABLES FOR WORKER PROCESSES ---
# These exist independently inside EACH worker process.
worker_task = None 
worker_agent = None

def init_worker(agent_class, starting_difficulty=0):
    """
    This runs ONCE when each worker process starts.
    `starting_difficulty` is passed via Pool initargs so the value is
    available even when workers are spawned as separate processes.
    """
    global worker_agent, worker_task, STARTING_DIFFICULTY
    STARTING_DIFFICULTY = starting_difficulty

    import multiprocessing
    worker_idx = int(multiprocessing.current_process().name.split('-')[-1]) - 1
    port = port_list[worker_idx % len(port_list)]

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

def get_pool(agent, population_size, starting_difficulty=0):
    global worker_pool
    if worker_pool is None:
        n_processes = min(N_PROCESSES, max(1, population_size))
        worker_pool = Pool(
            processes=n_processes,
            initializer=init_worker,
            initargs=(agent, starting_difficulty),
        )
    return worker_pool

def evaluate_population(agent, population, starting_difficulty=0):
    pool = get_pool(agent, len(population), starting_difficulty)
    rewards_list = pool.map(evaluate_individual, population)

    # We do NOT close the pool here anymore so workers persist across iterations.
    return np.array(rewards_list)