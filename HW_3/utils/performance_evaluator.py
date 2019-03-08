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
    worlds_number = 3
    worlds = list(world_generator(generate_map()) for _ in range(worlds_number))

    # create agents with all possible hyperparameters
    agents = []
    for (eta, reg_coef, epochs, reward_depth, train_every) \
            in list(itertools.product(
            # etas
            [0.00000001],
            # reg_coefs
            [0.01, 30],
            # epochs
            [60],
            # reward_depth
            [7],
            # train_every
            [25]
    )):
        if file is None:
            print("Creating a new agent")
            agent = SimpleCarAgent()
        else:
            print(f"Using an agent with weights from {file}")
            agent = SimpleCarAgent.from_file(file)
        if eta is not None:
            agent.eta = eta
        if reg_coef is not None:
            agent.reg_coef = reg_coef
        if epochs is not None:
            agent.epochs = epochs
        if reward_depth is not None:
            agent.reward_depth = reward_depth
        if train_every is not None:
            agent.train_every = train_every
        agents += [agent]

    errors = []

    for world in worlds:
        errors += [run_agent_for_worlds(agents, world, steps)]

    means = np.nanmean(errors, 0)[0]
    results = dict(zip(agents, means))

    best_agent = max(results, key=results.get)
    best_reward = results[best_agent]
    if type(best_reward) is not np.float64:
        best_reward = None

    for agent, result in results.items():
        print(f"Creating an agent with hyperparams: \n{agent.hyperparams_to_string()} \nError: {result}\n")

    print(f"\n🏆 This agent performed the best in all worlds with the error {best_reward}\n{best_agent.hyperparams_to_string()}")

    # write results to files
    file_path = str(file)
    dot_index = file_path.find(".")
    reward_file = file[:dot_index] + latest_error_file_suffid + file[dot_index:]
    with open(reward_file, 'a+') as f:
        f.seek(0)
        lines = f.readlines()
        last_reward = lines[-1] if len(lines) > 0 else None
        if best_reward is not None and (file is None or last_reward is None or float(last_reward) < float(best_reward)):
            save_to_file(best_agent)
            if last_reward is not None:
                f.write('\n')
            f.write(str(best_reward))
        else:
            print(f"Reward ({best_reward}) was invalid or worse than {last_reward} and wasn\'t saved")


def run_agent_for_worlds(agents, world, steps):
    rewards = []
    world.set_agents(agents)
    try:
        rewards += [world.run(steps)]
    except AssertionError:
        rewards += [[np.nan] * len(agents)]
    return rewards
