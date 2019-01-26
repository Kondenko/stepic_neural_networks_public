# from HW_3.cars import *
from cars.world import SimpleCarWorld
from cars.agent import SimpleCarAgent
from cars.physics import SimplePhysics
from cars.track import generate_map
import numpy as np
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--steps", type=int)
parser.add_argument("-f", "--filename", type=str)
parser.add_argument("-e", "--evaluate", type=bool)
parser.add_argument("-v", "--visualise", action="store_true")
parser.add_argument("--seed", type=int)
args = parser.parse_args()

# print(args.steps, args.seed, args.filename, args.evaluate, args.visualise)

steps = args.steps
seed = args.seed if args.seed else None # Use a random map
np.random.seed(seed)
random.seed(seed)
m = generate_map(8, 5, 3, 3)
visual = args.visualise

if args.filename:
    agent = SimpleCarAgent.from_file(args.filename)
    w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2)
    if args.evaluate:
        print(f"Evaluating on seed {seed}...")
        mean_error = w.evaluate_agent(agent, steps, visual)
    else:
        w.set_agents([agent])
        mean_error = w.run(steps)
    print(f"Error: {mean_error}")
else:
    SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2).run(steps)
