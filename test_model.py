import gymnasium as gym
import torch
from gymnasium.spaces import flatdim

from network import Network

if __name__ == "__main__":
    env_name = "LunarLander-v2"
    model_type = "ppo"
    model_name = "ppo_actor"
    model_layers = [256, 256, 256]
    continuous = True

    if continuous:
        env = gym.make(env_name, continuous=True, render_mode="human")
    else:
        env = gym.make(env_name, render_mode="human")

    obs_dims = flatdim(env.observation_space)
    act_dims = flatdim(env.action_space)

    model_path = f"models/{model_type}/{env_name}/{model_name}.pth"

    policy = Network(obs_dims, act_dims, model_layers)
    policy.load_state_dict(torch.load(model_path))

    while True:
        obs, _ = env.reset()

        for _ in range(200):
            action = policy(obs).detach().numpy()
            obs, reward, terminated, _, _ = env.step(action)

            if terminated:
                break
