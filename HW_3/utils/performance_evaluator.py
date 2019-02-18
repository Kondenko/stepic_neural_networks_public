import itertools
import random

import numpy as np

from cars.agent import SimpleCarAgent
from cars.track import generate_map
from utils.funcs import save_to_file


def run_and_save_best(world_generator, steps, file=None):
    np.random.seed(None)
    random.seed(None)
    worlds = (world_generator(generate_map()) for _ in range(3))
    agent_reward_pairs = (run_agents_for_world(w, steps, file) for w in worlds)
    best_agent, best_reward = max(agent_reward_pairs, key=lambda pair: pair[1])
    print(f"The agent with eta={best_agent.eta}, reg_coef={best_agent.reg_coef} performed the best in all worlds:\n{best_reward}")
    save_to_file(best_agent)


def run_agents_for_world(world, steps, file=None) -> tuple:
    """
    Trains multiple networks with different hyperparameters, chooses the network
    with the best result and saves in to a file. 
    
    :param generate_world:
    :type world: SimpleCarWorld
    """
    etas = [0.02, 0.05, 0.1]
    reg_coefs = [0.0001, 0.001, 0.01]
    product = list(itertools.product(etas, reg_coefs))
    agents = []
    for (eta, reg_coef) in product:
        if file is None:
            print("Creating a new agent")
            agent = SimpleCarAgent()
        else:
            print("Using an agent from file")
            agent = SimpleCarAgent.from_file(file)
        agent.set_hyperparams(eta=eta, reg_coef=reg_coef)
        agents += [agent]
    world.set_agents(agents)
    mean_rewards = world.run(steps)
    best_agent, best_reward = max(zip(agents, mean_rewards), key=lambda zipped: zipped[1])  # choose max reward
    print(f"The agent with eta={best_agent.eta}, reg_coef={best_agent.reg_coef} performed the best in world {world}:\n{best_reward}")
    return best_agent, best_reward
