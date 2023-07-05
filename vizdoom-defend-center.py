"""
Doom Ai


"""


# Getting Vizdoom up and running

from vizdoom import DoomGame
# Import random for action sampling
import random
# Import time for sleeping
import time
# Import numpy for identity matrix
import numpy as np

# Setup game
game = DoomGame()
game.load_config('Github/ViZDoom/scenarios/defend_the_line.cfg')
game.init()

# Creating some actions that our ai can take

actions = np.identity(3, dtype=np.uint8)

# The random choice here is purely for demonstration, eventually this logic will be replaced with actions selected by
# the PPO model

random.choice(actions)



game.new_episode()
game.is_episode_finished()
game.close()
state = game.get_state()
state.screen_buffer
game.make_action(random.choice(actions))
state.game_variables

game.make_action()



episodes = 10
for episode in range(episodes):
    # Create new episode
    game.new_episode()
    # check that game is not done
    while not game.is_episode_finished():
        # Get the game state
        state = game.get_state()
        # Get the game image
        img = state.screen_buffer
        # Get the game variables - ammo
        info = state.game_variables
        # take an action
        reward = game.make_action(random.choice(actions),4)
        # Print reward
        print('reward:', reward)
        time.sleep(0.02)
    print('Result', game.get_total_reward())
    time.sleep(2)

game.close()

# Import environment base class from OpenAI Gym
from gym import Env
# Import gym spaces
from gym.spaces import Discrete, Box
# Import opencv
import cv2


game.get_state().screen_buffer.shape


# Create Vizdoom OpenAI Gym Environment
class VizDoomGym(Env):
    # Function that is called when we start the env
    def __init__(self, render=False):
        # Inherit from Env
        super().__init__()
        # Setup the game
        self.game = DoomGame()
        self.game.load_config('Github/ViZDoom/scenarios/defend_the_line.cfg')

        # Render frame logic
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        # Start the game
        self.game.init()

        # Create the action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(3)

    # This is how we take a step in the environment
    def step(self, action):
        # Specify action and take step
        actions = np.identity(3)
        reward = self.game.make_action(actions[action], 4)

        # Get all the other stuff we need to retun
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info": info}
        done = self.game.is_episode_finished()

        return state, reward, done, info

        # Define how to render the game or environment

    @staticmethod
    def render():
        pass

    # What happens when we start a new game
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)

    # Grayscale the game frame and resize it
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state

    # Call to close down the game
    def close(self):
        self.game.close()

env = VizDoomGym(render=True)

state = env.reset()
env.close()

#3. View Game State

from stable_baselines3.common import env_checker
env_checker.check_env(env)

# 3. View State

from matplotlib import pyplot as plt
plt.imshow(cv2.cvtColor(state, cv2.COLOR_BGR2RGB))

# 4. Setup Callback

# Import os for file nav
import os
# Import callback class from sb3
from stable_baselines3.common.callbacks import BaseCallback


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


CHECKPOINT_DIR = './train/train_defend'
LOG_DIR = './logs/log_defend'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# 5. Train Model

# import ppo for training
from stable_baselines3 import PPO


# Non rendered environment
env = VizDoomGym()


model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)
# model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1)



model.learn(total_timesteps=100000, callback=callback)


# 6. Test the Model

# Import eval policy to test agent
from stable_baselines3.common.evaluation import evaluate_policy

# Reload model from disc
model1 = PPO.load('./train/train_defend/best_model_50000')

# Create rendered environment
env = VizDoomGym(render=True)

# Evaluate mean reward for 10 games
mean_reward, _ = evaluate_policy(model1, env, n_eval_episodes=100)



for episode in range(5):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model1.predict(obs)
        obs, reward, done, info = env.step(action)
        # time.sleep(0.20)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))
    time.sleep(2)

env.close()
