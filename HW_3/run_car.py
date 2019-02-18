# from HW_3.cars import *
import argparse
import os.path
import random

import numpy as np

from cars.agent import SimpleCarAgent
from cars.physics import SimplePhysics
from cars.track import generate_map
from cars.world import SimpleCarWorld
from utils.performance_evaluator import run_and_save_best

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--steps", type=int)
parser.add_argument("-f", "--filename", type=str)
parser.add_argument("-e", "--evaluate", type=bool)
parser.add_argument("-v", "--visualise", action="store_true")
parser.add_argument("-fb", "--findbest", action="store_true")
parser.add_argument("--seed", type=int)
args = parser.parse_args()

# print(args.steps, args.seed, args.filename, args.evaluate, args.visualise)

steps = args.steps
seed = args.seed if args.seed else None  # Use a random map
np.random.seed(seed)
random.seed(seed)
m = generate_map(8, 5, 3, 3)
visual = args.visualise
find_best = args.findbest


def train(world, steps):
    if find_best:
        run_and_save_best(lambda map: SimpleCarWorld(1, map, SimplePhysics, SimpleCarAgent, visual, timedelta=0.2), steps)
    else:
        w.set_agents([agent])
        w.run(steps)


if args.filename and os.path.isfile(args.filename):
    agent = SimpleCarAgent.from_file(args.filename)
    w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, visual, timedelta=0.2)
    if args.evaluate:
        print(f"Evaluating on seed {seed}...")
        print(f"Error: {w.evaluate_agent(agent, steps)}")
    else:
        train(w, steps)
else:
    train(SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, visual, timedelta=0.2), steps)
