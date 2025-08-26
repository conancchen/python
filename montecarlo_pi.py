import numpy as np
import random
import time

# Estimating Pi with a Monte Carlo Simulation
# Method 1: Circle in Square
# generate a bunch of random points and see if they lie within a circle
# points in circle / total points should = pi / 4

class CircleInSquare:
    def __init__(self, sims):
        self.sims = sims
        if sims > 1e8 + 1:
            raise ValueError("My computer not strong like that")

    def estimate(self):
        # define inside as # of points inside
        start = time.time()
        inside = 0
        for _ in range(1, self.sims):
            x1 = random.random()
            y1 = random.random()
            distanceSquared = x1 ** 2 + y1 ** 2
            if distanceSquared < 1:
                inside += 1
        end = time.time()
        return 4 * inside / self.sims, end - start

    def fastEstimate(self):
        start = time.time()
        x = np.random.rand(self.sims)
        y = np.random.rand(self.sims)
        inside = np.sum(x**2 + y**2 < 1)
        end = time.time()
        return inside * 4 / self.sims, end - start


# Try it

piEstimate1 = CircleInSquare(int(1e8))
# print(piEstimate1.estimate()[0], "Computation Time (s):", round(piEstimate1.estimate()[1], 5))
print(piEstimate1.fastEstimate()[0], "Compuation Time (s):", round(piEstimate1.fastEstimate()[1], 5))


# Method 2: Buffon's Needle Problem





        
