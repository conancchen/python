import numpy as np
import math
import random
import matplotlib.pyplot as plt

class RandomWalk:
    # default 
    def __init__(self, start, upper, lower, steps, stepSize = 1, pUp = 0.5):
        self.start = start
        self.upper = upper
        self.lower = lower
        self.steps = steps
        self.stepSize = stepSize
        self.pUp = pUp
        self.pDown = 1 - pUp

        if self.start < self.lower or self.start > self.upper:
            raise ValueError("Start position must be within the bounds of lower and upper.")

        if pUp < 0 or pUp > 1:
            raise ValueError("Probability of moving up must be between 0 and 1.")
        
    # path-generating functions
    def generatePath(self):
        path = [self.start]
        for _ in range(self.steps):
            step = np.random.choice(
                [self.stepSize, -self.stepSize],
                p=[self.pUp, self.pDown]
            )
            path.append(path[-1] + step) # adds step to the last element of path array
        
        return path
    
    # this function returns the path up until the hit
    def gamblersRuin(self):
        path = [self.start]
        for _ in range(self.steps):
            step = np.random.choice([self.stepSize, -self.stepSize], p=[self.pUp, self.pDown])
            path.append(path[-1] + step) # adds step to the last element of path array

            # Check for gambler's ruin conditions
            if path[-1] >= self.upper or path[-1] <= self.lower:
                break
        return path

    # some properties of the random walks
    def hittingTime(self):
        path = self.gamblersRuin()
        return len(path) - 1  # Subtract 1 to exclude the starting position

    def expectedHT(self):
        return (self.start - self.lower) * (self.upper - self.start)
    
    def endPoint(self):
        path = self.generatePath()
        return path[-1]
    
    def plotPath(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.generatePath(), marker='o')
        plt.title('Random Walk Path')
        plt.xlabel('Steps')
        plt.ylabel('Position')
        plt.axhline(y=self.upper, color='r', linestyle='--', label='Upper Bound')
        plt.axhline(y=self.lower, color='g', linestyle='--', label='Lower Bound')
        plt.legend()
        plt.grid()
        plt.show()
    
    def plotHT(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.gamblersRuin(), marker='o')
        plt.title('Random Walk Path')
        plt.xlabel('Steps')
        plt.ylabel('Position')
        plt.axhline(y=self.upper, color='r', linestyle='--', label='Upper Bound')
        plt.axhline(y=self.lower, color='g', linestyle='--', label='Lower Bound')
        plt.legend()
        plt.grid()
        plt.show()

    def plotHTs(self, num_trials=1000):
        hitting_times = []
        for _ in range(num_trials):
            hitting_times.append(self.hittingTime())
        
        plt.figure(figsize=(10, 5))
        plt.hist(hitting_times, bins=30, alpha=0.7, color='blue')
        plt.title('Distribution of Hitting Times')
        plt.xlabel('Hitting Time')
        plt.ylabel('Frequency')
        plt.grid()
        plt.show()

    def plotEndPoint(self, num_trials=1000):
        end_points = []
        for _ in range(num_trials):
            end_points.append(self.generatePath()[-1])
        
        plt.figure(figsize=(10, 5))
        plt.hist(end_points, bins=30, alpha=0.7, color='blue')
        plt.title('Distribution of End Points')
        plt.xlabel('End Point')
        plt.ylabel('Frequency')
        plt.grid()
        plt.show()

    
    def changeStepSize(self, newStepSize):
        self.stepSize = newStepSize
        self.path = self.generatePath()
    
    def changeProbabilities(self, pUp):
        if not 0 <= pUp <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        
        self.pUp = pUp
        self.pDown = 1 - pUp

        self.path = self.generatePath()

    def changeStartPoint(self, newStart):
        if not self.lower <= newStart <= self.upper:
            raise ValueError("Start point must be between boundaries.")
        
        self.start = newStart
        self.path = self.generatePath()
    
    def changeBounds(self, newUpper, newLower):
        if newUpper <= newLower:
            raise ValueError("Upper bound must be greater than lower bound.")
        
        self.upper = newUpper
        self.lower = newLower
        
        if not self.lower <= self.start <= self.upper:
            raise ValueError("Start point must be within the new bounds.")
        
        self.path = self.generatePath()
    
    def reset(self):
        # reset to simple symmetric random walk w/ default parameters
        self.start = 0
        self.upper = 100
        self.lower = -100
        self.steps = 1000
        self.stepSize = 1
        self.pUp = 0.5
        self.pDown = 0.5
        print("Random walk has been reset to initial conditions.")
    
    def __str__(self):
        return (f"Random Walk:\n"
                f"Start: {self.start}\n"
                f"Upper Bound: {self.upper}\n"
                f"Lower Bound: {self.lower}\n"
                f"Steps: {self.steps}\n"
                f"Step Size: {self.stepSize}\n"
                f"Probability of Upward Step: {self.pUp}")


'''# Example usage
if __name__ == "__main__":
    rw1 = RandomWalk(10, 20, 0, 100)
    rw1.generatePath()
    rw1.plotHT()'''

'''class BrownianMotion(RandomWalk):
    def __init__(self, start, upper, lower, time, mu=0, sigma=1, n=1000):
        super().__init__(start, upper, lower, steps=0, pUp = 0.5)
        self.pDown = 1 - pUp
        self.mu = mu
        self.sigma = sigma
        self.time = time
        self.N = n

        if self.time <= 0:
            raise ValueError("Time must be positive.")

        if n <= 0:
            raise ValueError("N must be a positive integer.")

    
    def generatePath(self):
        # generate +/- 1 random walks
        self.steps = N
        self.stepSize = 1
        rwPath = super().generatePath()
        increments = np.diff(rwPath) 
        # convert increments to brownian potion
        dt = self.time / N
        bmIncr = self.mu * dt + self.sigma * np.sqrt(dt) * np.array(increments) #don't rlly understand
        bm_path = np.cumsum(bmIncr)  # cumulative sum to get the path
        # add start
        bm_path = np.insert(bm_path, 0, self.start)
        time_grid = np.linspace(0, self.time, N + 1)
        return time_grid, bm_path
    

    def plotPath(self, time_grid, path):
        plt.figure(figsize=(10, 5))
        plt.plot(time_grid, path, marker='o', markersize=2)
        plt.title('Brownian Motion Path (from Random Walk)')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.axhline(y=self.upper, color='r', linestyle='--', label='Upper Bound')
        plt.axhline(y=self.lower, color='g', linestyle='--', label='Lower Bound')
        plt.legend()
        plt.grid()
        plt.show()

    def hittingTime(self, time_grid, path):
        for i, pos in enumerate(path):
            if pos >= self.upper or pos <= self.lower:
                return time_grid[i]
        return None  # if no hitting time


# Example usage
if __name__ == "__main__":
    bm = BrownianMotion(start=0, upper=100, lower=-100, time=10000, mu=0, sigma=1)
    bm.plotPath(time_grid=bm.generatePath()[0], path=bm.generatePath()[1])
    print(bm.hittingTime(time_grid=bm.generatePath()[0], path=bm.generatePath()[1]))'''


class StandardBrownian:
    # standard brownian walk: sigma = 1
    # epsilon = sigma / N (take n to infinity)
    # we need time, and we n (bombardments per unit time)
    def __init__(self, start, n, t, sigma = 1, pUp = 0.5, upper=None, lower=None):
        self.start = start
        self.n = n
        self.t = t
        self.totalMoves = n * t
        self.dt = 1 / self.n # for every step, we jump up 1/n units of time

        self.sigma = sigma
        self.epsilon = sigma / np.sqrt(n) #epsilon (displacement per step)

        self.pUp = pUp
        self.pDown = 1 - pUp

        self.upper = upper
        self.lower = lower

        if sigma < 0:
            raise ValueError("Sigma must be positive")
        if t < 0: 
            raise ValueError("Negative Time entered")
        if n < 0:
            raise ValueError("n must be positive")
        if not 0 <= pUp <= 1:
            raise ValueError("Probability must be between 0 and 1")

        if self.lower is not None and self.upper is not None:
            if not self.lower <= self.start <= self.upper:
                raise ValueError("Start must be between lower and upper bounds.")

        
    # creates a set path and time grid
    def move(self):
        path = [self.start]
        timeGrid = [0]
        # make the path
        for _ in range(1, self.totalMoves + 1):
            step = np.random.choice([self.epsilon, -self.epsilon], p=[self.pUp, self.pDown])
            path.append(path[-1] + step) # add the step to the last element of path
            timeGrid.append(timeGrid[-1] + self.dt) # adds increment
        
        self.timeGrid = timeGrid
        self.path = path

        return np.array(timeGrid), np.array(path)
    
    # tied to move
    def endPoint(self):
        return self.path[-1] #last element of path
    
    # plots the path
    def plot(self):
        # Ensure path and timeGrid exist
        if not hasattr(self, 'path') or not hasattr(self, 'timeGrid'):
            self.move()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.timeGrid, self.path, label='Path')

        # Draw bounds if set
        if self.upper is not None:
            ax.axhline(self.upper, color='r', linestyle='--', label='Upper Bound')
        if self.lower is not None:
            ax.axhline(self.lower, color='g', linestyle='--', label='Lower Bound')

        # Determine and mark hitting point
        hit = self.ht()  # uses self.upper and self.lower internally
        if hit.startswith("Hits"):
            # parse time
            t_str = hit.split("time")[1].strip()
            t_hit = float(t_str)
            # find index i where timeGrid ≈ t_hit
            idx = next(i for i, t in enumerate(self.timeGrid) if abs(t - t_hit) < 1e-8)
            # choose marker color
            color = 'r' if "Upper" in hit else 'g'
            # plot marker
            ax.plot(self.timeGrid[idx],
                    self.path[idx],
                    marker='o',
                    color=color,
                    markersize=8,
                    label=hit)

        ax.set_xlabel('Time')
        ax.set_ylabel('Position')
        ax.set_title('Standard Brownian Motion (Discrete Approximation)')
        ax.legend()
        ax.grid(True)
        plt.show()


    
    # helper method for plot endpoints
    def efficientEndpoint(self):
        endpoint = self.start
        for _ in range(1, self.totalMoves + 1):
            step = np.random.choice([self.epsilon, -self.epsilon], p=[self.pUp, self.pDown])
            endpoint += step
        return endpoint
    
    # generates and plots endpoints (distribution)
    def plotEndpoints(self, simulations, return_ax=False):
        self._endpoints = [self.efficientEndpoint() for _ in range(simulations)]
        self._endpoints = np.array(self._endpoints)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(self._endpoints, bins=30, density=True, alpha=0.6, color='blue', label='Simulated')
        ax.set_title('Distribution of Final Positions')
        ax.set_xlabel('Final Position')
        ax.set_ylabel('Density')
        ax.grid(True)

        # if return, return, if not dont
        if return_ax:
            return ax
        else:
            ax.legend()
            plt.show()

    # vibecoded, idk
    def overlayNormal(self, ax=None):
        if not hasattr(self, '_endpoints'):
            raise ValueError("Run plotEndpoints(simulations) first to generate endpoints.")

        mu = self.start
        std = self.sigma * np.sqrt(self.t)

        # Normal distribution PDF manually using NumPy
        x = np.linspace(min(self._endpoints), max(self._endpoints), 500)
        pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)

        # Plot on given axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(self._endpoints, bins=30, density=True, alpha=0.6, color='blue', label='Simulated')

        ax.plot(x, pdf, 'r--', label=f'Normal(μ={mu}, σ²={round(std**2, 2)})')
        ax.legend()
        plt.show()
    
    # CDF
    def cdf(self, x):
        mu = self.start
        std = self.sigma * np.sqrt(self.t)
        z = (x - mu) / (std * np.sqrt(2))
        return 0.5 * (1 + math.erf(z))
    
    # probability the brownian motion ends beween a and b
    def endingProbability(self, lower, upper):
        return self.cdf(upper) - self.cdf(lower)
    

    # ------------ HITTING TIMES -----------------------

    def ht(self):
        if not hasattr(self, 'path') or not hasattr(self, 'timeGrid'):
            self.move()

        for i in range(1, len(self.path)):  # <-- start from 1
            pos = self.path[i]
            if self.upper is not None and pos >= self.upper:
                return f"Hits Upper Bound at time {round(self.timeGrid[i], 4)}"
            elif self.lower is not None and pos <= self.lower:
                return f"Hits Lower Bound at time {round(self.timeGrid[i], 4)}"

        return "Does not hit"
    
    def efficientHT(self):
        position = self.start
        steps = 0
        hitTime = 0.0

        # Keep jumping until we cross a bound
        while True:
            # take one ε‐step
            r = np.random.rand()
            step = self.epsilon if r < self.pUp else -self.epsilon

            position += step
            hitTime += self.dt

            # check upper bound
            if self.upper is not None and position >= self.upper:
                return f"Hits Upper Bound at time {round(hitTime, 4)}"

            # check lower bound
            if self.lower is not None and position <= self.lower:
                return f"Hits Lower Bound at time {round(hitTime, 4)}"

    def whichBoundary(self):
        position = self.start
        steps = 0
        hitTime = 0.0

        # Keep jumping until we cross a bound
        while True:
            # take one ε‐step
            step = np.random.choice([self.epsilon, -self.epsilon], p=[self.pUp, self.pDown])
            position += step
            hitTime += self.dt

            # check upper bound
            if self.upper is not None and position >= self.upper:
                return "Up"

            # check lower bound
            if self.lower is not None and position <= self.lower:
                return "Low"

    # determines how often we hit the upper bound
    # pretty fucking slow man
    def winRate(self, simulations):
        count = 0
        for _ in range(simulations):
            if self.whichBoundary() == "Up":
                count += 1
        return round(count / simulations, 5) 


sbm = StandardBrownian(start=0, n=1000, t=30, pUp = 0.52, upper = 100, lower = -10)

sbm.move()
sbm.plot()
sbm.winRate(10)
