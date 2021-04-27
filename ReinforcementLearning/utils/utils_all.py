import numpy as np
import matplotlib.pyplot as plt


# A buffer used for replay memory
class Buffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = np.array([None for _ in range(max_size)])
        self.point = 0

    def sample_batch(self, batch_size):
        if self.buffer[-1] is None:
            choose_idx = np.random.choice(self.point, batch_size)
        else:
            choose_idx = np.random.choice(self.max_size, batch_size)
        return self.buffer[choose_idx]

    def append(self, record):
        # record: tuple
        self.buffer[self.point] = record
        self.point += 1
        if self.point >= self.max_size:
            self.point %= self.max_size


# Roll average class
class RollAverage(object):
    def __init__(self, window=10):
        self.window = window
        self.total = 0
        self.stack = []
        self.record = []

    def update(self, number):
        self.stack.append(number)
        if len(self.stack) <= self.window:
            self.total += number
            res = self.total / len(self.stack)
        else:
            old = self.stack.pop(0)
            self.total += number - old
            res = self.total / self.window
        self.record.append(res)
        return res

    def show(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.record)
        plt.xlabel("Iteration", fontsize=16)
        plt.ylabel("Total Reward", fontsize=16)
        plt.show()


