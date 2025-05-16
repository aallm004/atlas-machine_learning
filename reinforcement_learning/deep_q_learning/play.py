#!/usr/bin/env python3
"""Module to play Breakout using a trained DQN agent"""

import numpy as np
import gymnasium as gym
import time

# Same patching and imports as training
import sys
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras
sys.modules['tensorflow.keras'] = keras

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor
from gymnasium.wrappers import AtariPreprocessing

class AtariProcessor(Processor):
    """Custom processor for Atari environments"""
    
    def process_observation(self, observation):
        """process observation"""
        return observation
        
    def process_state_batch(self, batch):
        """Process batch of states"""
        batch = np.squeeze(batch, axis=-1)
        batch = np.transpose(batch, (0, 2, 3, 1))
        return batch.astype('float32') / 255.0
        
    def process_reward(self, reward):
        """Process reward"""
        return np.clip(reward, -1., 1.)

class GymCompatibilityWrapper(gym.Wrapper):
    """Wrapper to make Gymnasium work with keras-rl2"""
    
    def reset(self, **kwargs):
        """Update reset method to match keras-rl expected output"""
        obs, info = self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        """Update step method to match keras-rl expected output"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def render(self, mode=None):
        """Update render method for compatibility
        Ignore the mode parameter that keras-rl2 will pass"""
        return self.env.render()

def build_model(input_shape, actions):
    """Build CNN model for DQN"""
    model = keras.Sequential()
    
    # convolutional layers
    model.add(keras.layers.Conv2D(32, (8, 8), strides=(4, 4),
                                  activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (4, 4), strides=(2, 2),
                                  activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1),
                                  activation='relu'))
    
    # fully connected layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(actions, activation='linear'))
    
    return model

def build_agent(model, actions):
    """Build a DQN agent"""
    memory = SequentialMemory(limit=1000000, window_length=4)
    
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), 
        attr='eps', 
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=1000000
    )
    
    processor = AtariProcessor()
    
    dqn = DQNAgent(model=model, 
                  memory=memory, 
                  policy=policy, 
                  processor=processor,
                  nb_actions=actions,
                  nb_steps_warmup=50000, 
                  target_model_update=10000,
                  enable_double_dqn=True)
    
    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.00025)
    dqn.compile(optimizer)
    
    return dqn

def main():
    """Main function to play using the trained agent."""
    print("Creating Breakout environment...")
    env = gym.make('ALE/Breakout-v5', frameskip=1, render_mode='human')
    
    print("Applying preprocessing...")
    # Apply the SAME preprocessing as during training
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=True,
        scale_obs=False
    )
    
    # Aapply compatibility wrapper
    env = GymCompatibilityWrapper(env)
    
    # Setup with SAME dimensions as training
    input_shape = (84, 84, 4)
    actions = env.action_space.n
    
    print("Building model...")
    # Nuild the model with the SAME architecture
    model = build_model(input_shape, actions)
    
    print("Building agent...")
    # Build agent
    dqn = build_agent(model, actions)
    
    print("Loading weights from policy.h5...")
    # load the weights from training
    dqn.load_weights('policy.h5')
    
    print("Starting game with trained agent...")
    # Test mode uses the test_policy_eps for less exploration/use learned
    try:
        dqn.test(env, nb_episodes=5, visualize=True)
    except Exception as e:

        manual_play(dqn, env)
    
    print("Game complete. Closing environment.")
    env.close()

def manual_play(dqn, env):
    """Manual play loop as backup in case dqn.test fails"""
    obs = env.reset()
    done = False
    total_reward = 0
    
    # Play for 5 episodes
    for episode in range(5):
        print(f"Starting episode {episode+1}")
        done = False
        episode_reward = 0
        step = 0
        
        while not done:
            # Get action from DQN
            action = dqn.forward(obs)
            
            # Take action in environment
            next_obs, reward, done, info = env.step(action)
            
            # Update for next step
            obs = next_obs
            episode_reward += reward
            step += 1
            
            # Possibly? slow down visualization
            time.sleep(0.01)
            
            # Print occasional progress
            if step % 100 == 0:
                print(f"Episode {episode+1}, Step {step}, "
                      f"Reward so far: {episode_reward}")
        
        # Episode complete
        print(f"Episode {episode+1} finished with total reward: "
              f"{episode_reward} after {step} steps")
        total_reward += episode_reward
        
        # Reset for next episode
        obs = env.reset()
    
    print(f"Played 5 episodes with average reward: {total_reward/5}")

if __name__ == "__main__":
    main()
