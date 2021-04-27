import time
import game.flappy_bird.wrapped_flappy_bird as fb
import matplotlib.pyplot as plt
import numpy as np

env = fb.GameState()
for _ in range(1000):
    p = np.random.rand()
    if p > 0.1:
        img, reward, done = env.frame_step([1, 0])
    else:
        img, reward, done = env.frame_step([0, 1])



