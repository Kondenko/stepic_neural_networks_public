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

steps = args.steps
seed = args.seed if args.seed else None  # Use a random map
np.random.seed(seed)
random.seed(seed)
m = generate_map(8, 5, 3, 3)
visual = args.visualise
file = args.filename
find_best = args.findbest


def train():
    if find_best:
        run_and_save_best(visual, steps, _map=m if seed is not None else None, file=file)
    else:
        w.set_agents([agent])
        w.run(steps)


if args.filename and os.path.isfile(file):
    agent = SimpleCarAgent.from_file(file)
    w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, visual, timedelta=0.2)
    if args.evaluate:
        print(f"Evaluating on seed {seed}...")
        print(f"Error: {w.evaluate_agent(agent, steps)}")
        print(f"Circles: {w.circles}")
    else:
        train()
else:
    train()
