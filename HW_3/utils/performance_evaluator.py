import itertools
from cars.agent import SimpleCarAgent


def run_and_save_best(world, steps, file=None):
    """
    Trains multiple networks with different hyperparameters, chooses the network
    with the best result and saves in to a file. 
    
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
    best_agent, best_reward = max(zip(agents, mean_rewards), key=lambda zipped: zipped[1]) # choose max reward
    world.save_to_file(best_agent)
    print(f"The best reward is \n{best_reward}")
