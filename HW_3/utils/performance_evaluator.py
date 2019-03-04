import itertools
import random

import numpy as np

from cars.agent import SimpleCarAgent
from cars.track import generate_map
from utils.funcs import save_to_file

latest_error_file_suffid = "_rewards"

def run_and_save_best(world_generator, steps, file=None):
    np.random.seed(None)
    random.seed(None)
    attempts = 7
    worlds = (world_generator(generate_map()) for _ in range(attempts))
    results = []
    for w in worlds:
        results += [run_agents_for_world(w, steps, file)]
    best_agent, best_reward = max(results, key=lambda pair: pair[1])
    print(f"The agent with eta={best_agent.eta}, reg_coef={best_agent.reg_coef} performed the best in all worlds:\n{best_reward}")
    file_path = str(file)
    dot_index = file_path.find(".")
    reward_file = file[:dot_index] + latest_error_file_suffid + file[dot_index:]
    with open(reward_file, 'a+') as f:
        lines = f.readlines()
        last_reward = lines[-1] if len(lines) > 0 else None
        if file is None or last_reward is None or last_reward > best_reward:
            save_to_file(best_agent)
            f.write(str(best_reward) + "\n")

def run_agents_for_world(world, steps, file=None):
    """
    Trains multiple networks with different hyperparameters, chooses the network
    with the best result and saves in to a file. 
    
    :param file: File
    :type world: SimpleCarWorld
    """
    etas = [0.000001, 0.0000005]
    reg_coefs = [0.001]
    product = list(itertools.product(etas, reg_coefs))
    agents = []
    for (eta, reg_coef) in product:
        if file is None:
            print("Creating a new agent")
            agent = SimpleCarAgent()
        else:
            print(f"Using an agent with weights from {file}")
            agent = SimpleCarAgent.from_file(file)
        agent.set_hyperparams(eta=eta, reg_coef=reg_coef)
        agents += [agent]
    world.set_agents(agents)
    try:
        mean_rewards = world.run(steps)
    except AssertionError:
        mean_rewards = None
    best_agent, best_reward = max(zip(agents, mean_rewards), key=lambda zipped: zipped[1])  # choose max reward
    print(f"The agent with eta={best_agent.eta}, reg_coef={best_agent.reg_coef} performed the best in world {world}:\n{best_reward}")
    return best_agent, best_reward
