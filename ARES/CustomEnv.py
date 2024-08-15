import numpy as np
from gymnasium import spaces, Env
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import pyautogui

import HadesController

pyautogui.FAILSAFE = False

class Hades(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, timelimit=75000, screen_size=None):
        super(Hades, self).__init__()
        if not screen_size:
            w, h = list(pyautogui.size())
            self.screen_size = [h, w, 3]
        else:
            self.screen_size = screen_size
        print(self.screen_size)

        self.action_space = spaces.MultiDiscrete([ 5, 4, 2, 200, 200 ])
        # self.action_space = spaces.Dict({"motion": spaces.Discrete(5), "attack": spaces.Discrete(4), "dash":spaces.Discrete(2), "cursor_velocity": spaces.Box(low=-100, high=100, shape=(2,), dtype=np.float16)})
        # self.observation_space = spaces.Tuple((spaces.Discrete(3), spaces.Box(low=0, high=255, shape=self.screen_size, dtype=np.uint8)))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.screen_size, dtype=np.uint8)
        self.timestep = 0
        self.timelimit = timelimit
        self.human_render = False
        self.returns = []
        self.episode_rewards = [0]

        self.screen_handler = HadesController.ScreenHandler()

    def step(self, action):
        done, trunc = False, False

        HadesController.move(direction = action[0])
        HadesController.attack(action = action[1])
        cusror_velocity = [action[3] - 100, action[4] - 100]
        print(cusror_velocity)
        HadesController.move_cursor(*cusror_velocity)
        if action[2]:
            HadesController.dash()
        
        obs = self.screen_handler.capture_screen()

        if self.screen_handler.get_end_of_room(obs):
            done = True

        info = {"": ""}
        self.timestep += 1

        if self.timelimit < self.timestep:
            print(self.timelimit, self.timestep)
            trunc = True

        if self.human_render:
            print(action)
            print(f"Total Reward: {np.sum(self.episode_rewards)}")
        
        # -1 reward for each timestep
        reward = -1

        self.episode_rewards.append(reward)
        return obs, reward, done, trunc, info

    def reset(self, seed=None):
        self.returns.append(np.sum(self.episode_rewards))
        if len(self.returns) % 50 == 0:
            plt.figure(figsize=[16, 12])
            plt.subplot(2, 1, 1)
            plt.plot(range(len(self.returns)), self.episode_rewards, label="rewards")
            plt.legend() 
            plt.subplot(2, 1, 2)
            plt.plot(range(len(self.returns)), self.returns, label="returns")
            plt.legend() 
            plt.savefig("2048_return_graph")
            plt.close()
        self.episode_rewards = [0]
        
        self.timestep = 0
        obs = self.screen_handler.capture_screen()
        return obs, {"": ""}  # reward, done, info can't be included
    
    def set_room_type(self, room_type):
        self.screen_handler.set_room_type(room_type)

    def render(self, mode='human'):
        if mode =="human":
            self.human_render = True

    def close (self):
        pass