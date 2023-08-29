import gymnasium as gym

import utils
from models import AgentPPO

if __name__ == "__main__":
    utils.set_global_seed(0)

    agent = AgentPPO(gym.make("LunarLander-v2", continuous=True), [64, 64])

    agent.learn(100_000)
