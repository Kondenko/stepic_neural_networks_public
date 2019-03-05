import itertools
import random

import numpy as np

from cars.agent import SimpleCarAgent
from cars.track import generate_map
from utils.funcs import save_to_file

latest_error_file_suffid = "_rewards"


def run_and_save_best(world_generator, steps, file=None):
    """
    Trains multiple networks with different hyperparameters, chooses the network
    with the best result and saves in to a file.

    :param file: File
    :type world: SimpleCarWorld
    """
    # create worlds to run the agents on
    np.random.seed(None)
    random.seed(None)
    attempts = 7
    worlds = list(world_generator(generate_map()) for _ in range(attempts))

    # create agents with all possible hyperparameters
    etas = [0.000001, 0.0000005]
    reg_coefs = [0.001]
    hyperparams_combinations = list(itertools.product(etas, reg_coefs))
    agents = []
    for (eta, reg_coef) in hyperparams_combinations:
        if file is None:
            print("Creating a new agent")
            agent = SimpleCarAgent()
        else:
            print(f"Using an agent with weights from {file}")
            agent = SimpleCarAgent.from_file(file)
        agent.set_hyperparams(eta=eta, reg_coef=reg_coef)
        agents += [agent]

    errors = []

    for world in worlds:
        errors += [run_agent_for_worlds(agents, world, steps)]

    results = dict(zip(agents, np.mean(errors, 0)[0]))

    best_agent = max(results, key=results.get)
    best_reward = results[best_agent]
    print(f"The agent with eta={best_agent.eta}, reg_coef={best_agent.reg_coef} performed the best in all worlds:\n{best_reward}")

    # write results to files
    file_path = str(file)
    dot_index = file_path.find(".")
    reward_file = file[:dot_index] + latest_error_file_suffid + file[dot_index:]
    with open(reward_file, 'a+') as f:
        f.seek(0)
        lines = f.readlines()
        last_reward = lines[-1] if len(lines) > 0 else None
        if file is None or last_reward is None or float(last_reward) < float(best_reward):
            save_to_file(best_agent)
            if last_reward is not None:
                f.write('\n')
            f.write(str(best_reward))


def run_agent_for_worlds(agents, world, steps):
    rewards = []
    world.set_agents(agents)
    try:
        rewards += [world.run(steps)]
    except AssertionError:
        rewards += [None]
    return rewards
