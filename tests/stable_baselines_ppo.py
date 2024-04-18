from __future__ import annotations

import argparse
import glob
import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy

from toolsrl.tools_raw_env import env, parallel_env, raw_env


def train_butterfly_supersuit(
    parallel_env, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = parallel_env(**env_kwargs)

    if env_kwargs["policy"] == "cnn":
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = PPO(
        MlpPolicy if env_kwargs["policy"] == "mlp" else CnnPolicy,
        env,
        verbose=3,
        learning_rate=1e-4,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)
    model.save(os.path.join("models", f"{env_kwargs['policy']}-{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"))
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    env.close()


def eval(env_constructor, num_games: int = 100, render_mode: str | None = None, deterministic: bool = False, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_constructor(render_mode=render_mode, **env_kwargs)

    if env_kwargs["policy"] == "cnn":
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})")

    try:
        latest_policy = max(
            glob.glob(os.path.join("models", f"{env_kwargs['policy']}-{env.metadata['name']}*.zip")), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}
    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=deterministic)[0]

            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--policy", default="cnn", choices=["mlp", "cnn"])
    args.add_argument("--deterministic", default="False", choices=["True", "False"])
    args.add_argument("--steps", default=409600, type=int)
    args.add_argument("--train", default="False", choices=["True", "False"])
    args = args.parse_args()
    env_kwargs = {"policy": args.policy}
    args.deterministic = True if args.deterministic == "True" else False
    args.train = True if args.train == "True" else False

    # Train a model
    if args.train:
        train_butterfly_supersuit(parallel_env, steps=args.steps, seed=41, **env_kwargs)

    # Evaluate 1 game
    # eval(env, num_games=1, render_mode=None, deterministic=args.deterministic, **env_kwargs)

    # Watch 2 games
    eval(env, num_games=2, render_mode="human", deterministic=args.deterministic, **env_kwargs)
