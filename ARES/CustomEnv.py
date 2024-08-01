import numpy as np
from gymnasium import spaces, Env
from gymnasium.envs.registration import register

import HadesController

class CustomEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, timelimit=750, screen_size=[960, 1080]):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Dict({"motion": spaces.Discrete(5), "attack": spaces.Discrete(4), "dash":spaces.Discrete(2), "cursor_velocity": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float16)})
        self.observation_space = spaces.Tuple((spaces.Discrete(3), spaces.Box(low=0, high=255, shape=screen_size, dtype=np.int8)))
        self.timestep = 0
        self.timelimit = timelimit
        self.human_render = False
        self.returns = []
        self.rewards = [0]

    def step(self, action):
        done, trunc = False, False

        HadesController.move(direction = action["motion"])
        HadesController.attack(action = action["attack"])
        print(action["cursor_velocity"])
        HadesController.move_cursor(action["cursor_velocity"][0], action["cursor_velocity"][1])
        if action["dash"]:
            HadesController.dash()
        


        status = logic.get_current_state(self.mat)
        if(status == 'GAME NOT OVER'):
            logic.add_new_2(self.mat)
            # zero for each move
            # reward = 0
        else:
            done = True
            # reward is the largest value at end of ep, or sum of all tiles
            # reward = np.max(self.mat)
            # reward = np.sum(self.mat)
        info = {"": ""}
        self.timestep += 1

        if self.timelimit < self.timestep:
            print(self.timelimit, self.timestep)
            trunc = True
            # reward is the largest value at end of ep, or sum of all tiles
            # reward = np.max(self.mat)
            # reward = np.sum(self.mat)

        if self.human_render:
            print(np.array(self.mat))
            print()
            time.sleep(0.25)
        
        # reward is the number of free tiles
        # reward += np.count_nonzero(self.mat)
        # the value of the merged numbers
        reward = merged

        self.rewards[-1] += reward
        # return np.array(self.mat).flatten(), reward, done, trunc, info
        obs = np.array(self.mat)
        obs = np.expand_dims(obs, axis=0)
        return obs, reward, done, trunc, info