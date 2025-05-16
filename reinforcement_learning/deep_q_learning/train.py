#!/usr/bin/env python3
"""
Module to train a DQN agent to play Atari's Breakout
"""

import numpy as np
import gymnasium as gym
import os

# Patch for rl library to use standalone keras not tf
import sys
import tensorflow as tf 
tf.compat.v1.disable_eager_execution()
import keras
sys.modules['tensorflow.keras'] = keras

# Import rl modules
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor
from gymnasium.wrappers import AtariPreprocessing


class AtariProcessor(Processor):
    """Custom processor for Atari environments."""
    
    def process_observation(self, observation):
        """Process observation"""
        # Return observation
        return observation
        
    def process_state_batch(self, batch):
        """Process batch of states.
            batch: Batch of observations with shape (batch_size,
            history_length, height, width, channels)
            
        returns: Processed batch with shape (batch_size, height, width,
        history_length)
        """
        # From (batch_size, history_length, height, width, channels) to
        # (batch_size, height, width, history_length)
        # First, remove the channels dimension (it's just 1)
        batch = np.squeeze(batch, axis=-1)
        
        # Transpose to get channels last format
        # From (batch_size, history_length, height, width) to
        # (batch_size, height, width, history_length)
        batch = np.transpose(batch, (0, 2, 3, 1))
        
        # Convert to float32 and normalize
        return batch.astype('float32') / 255.0
        
    def process_reward(self, reward):
        """Process reward."""
        # Clip rewards to [-1, 1]
        return np.clip(reward, -1., 1.)


class GymCompatibilityWrapper(gym.Wrapper):
    """Wrapper to make Gymnasium compatible with keras-rl2"""
    
    def reset(self, **kwargs):
        """Update reset method to match keras-rl expected output."""
        obs, info = self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        """Update step method to match keras-rl expected output"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Combine terminated and truncated into singular
        done = terminated or truncated
        return obs, reward, done, info
    
    def render(self):
        """Update render method for compatibility."""
        return self.env.render()


def build_model(input_shape, actions):
    """Build a CNN model for DQN.
        input_shape: input shape (height, width, channels)
        actions: number of possible actions
        
    Returns: a Keras Sequential model
    """
    model = keras.Sequential()
    
    # Convolutional layers
    model.add(keras.layers.Conv2D(32, (8, 8), strides=(4, 4),
                                  activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (4, 4), strides=(2, 2),
                                  activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1),
                                  activation='relu'))
    
    # Fully connected layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(actions, activation='linear'))
    
    return model


def build_agent(model, actions):
    """Build a DQN agent.
        model: Keras model
        actions: Number of possible actions
        
    Returns: a DQNAgent instance
    """
    # Set up memory for experience replay
    memory = SequentialMemory(limit=1000000, window_length=4)
    
    # Use epsilon-greedy policy for exploration with annealing
    # Anneling is the model's willingness to experiment/try new things at first
    # And then later apply what it's learned provides best outcome
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), 
        attr='eps', 
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=1000000
    )
    
    # Create processor
    processor = AtariProcessor()
    
    # Create the DQN agent
    dqn = DQNAgent(model=model, 
                  memory=memory, 
                  policy=policy, 
                  processor=processor,
                  nb_actions=actions,
                  nb_steps_warmup=50000,
                  target_model_update=10000,
                  enable_double_dqn=True)
    
    # Compile the agent with Adam optimizer
    # Use legacy optimizer for compatibility b/w tensforflow/keras and kerasrl2
    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.00025)
    dqn.compile(optimizer)
    
    return dqn


def main():
    """Main function to train the agent."""
    # Create Breakout environment with frame w frameskip/ agent only sees and
    # Makes decisions ever 4th frame
    env = gym.make('ALE/Breakout-v5', frameskip=1)
    
    # Gymnasium wrappers for preprocessing
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=True,
        scale_obs=False
    )
    
    # Wrapper for keras-rl2
    env = GymCompatibilityWrapper(env)
    
    # Get environment dimensions and input shape for model
    # Input shape is (height, width, stacked_frames)
    input_shape = (84, 84, 4)
    actions = env.action_space.n
    
    print(f"Environment shape: {env.observation_space.shape}")
    print(f"Input shape for model: {input_shape}")
    print(f"Number of actions: {actions}")
    
    # Build model and agent
    model = build_model(input_shape, actions)
    model.summary()
    dqn = build_agent(model, actions)
    
    # Train agent
    dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)
    
    # Save end to policy
    dqn.save_weights('policy.h5', overwrite=True)
    
    env.close()


if __name__ == "__main__":
    main()
