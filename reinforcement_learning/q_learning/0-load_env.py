#!/usr/bin/env python3
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Function that loads the pre-made FrozenLakeEnv enviroment from
    gynasium

    desc is either None or a list of lists containing a custom description of
    the map to load for the environment
    map_name is either None or a string containing the pre-made map to load
    NOTE: if both desc and map_name are None, the environment will load a
    randomly generated 8x8 map
    is_slippery is a boolean to determine if the ice is slippery

    Returns: the environment"""
    env = gym.make("FrozenLake-v1",
                   map_name=map_name,
                   desc=desc,
                   is_slippery=is_slippery)

    return env
