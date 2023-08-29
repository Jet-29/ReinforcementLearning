import gymnasium as gym

import utils
from models import AgentPPO

if __name__ == "__main__":
    utils.set_global_seed(0)

    hyperparameters = {
        'time_steps_per_batch': 2048,
        'max_time_steps_per_episode': 200,
        'gamma': 0.99,
        'updates_per_iteration': 10,
        'lr': 1e-4,
        'clip': 0.2,
    }

    agent = AgentPPO(gym.make("LunarLander-v2", continuous=True), [256, 256, 256], hyperparameters=hyperparameters)

    agent.learn(100_000_000)
