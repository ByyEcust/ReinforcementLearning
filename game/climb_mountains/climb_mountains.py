import copy
import matplotlib.pyplot as plt
import json


PUNISHMENT = 1e3


class ClimbMountain(object):
    def __init__(self, mountain_cfg):
        self.width = None
        self.length = None
        self.height = None
        self.start = None
        self.target = None
        self.build_env(mountain_cfg)
        self.current_loc = None
        self.reset()
        self.ax = plt.subplot()

    def build_env(self, mount_cfg):
        with open(mount_cfg) as f:
            cfg = json.load(f)
        self.width = cfg.get("width", None)
        self.length = cfg.get("length", None)
        self.height = cfg.get("height", None)
        self.start = cfg.get("start", None)
        self.target = cfg.get("target", None)

    def reset(self):
        self.current_loc = copy.copy(self.start)
        return self.get_state()

    def get_state(self):
        return copy.copy(self.current_loc)

    def step(self, action):
        # 1: y-1, 2: x-1, 3: y+1, 4: x+1
        state = [x, y] = self.current_loc
        assert action in [1, 2, 3, 4], "The action is not valid, please make sure your action is in [1, 2, 3, 4], " \
                                       "the current action is {}".format(action)
        if action == 1:
            if y-1 >= 0:
                state = [x, y-1]
            else:
                return [x, y], -PUNISHMENT, True
        elif action == 2:
            if x-1 >= 0:
                state = [x-1, y]
            else:
                return [x, y], -PUNISHMENT, True
        elif action == 3:
            if y+1 < self.width:
                state = [x, y+1]
            else:
                return [x, y], -PUNISHMENT, True
        elif action == 4:
            if x+1 < self.length:
                state = [x+1, y]
            else:
                return [x, y], -PUNISHMENT, True

        self.current_loc = state
        done = state == self.target
        return copy.copy(state), 0 if done else -self.height[state[1]][state[0]], done

    def state2num(self, state):
        return state[0] * self.width + state[1]





