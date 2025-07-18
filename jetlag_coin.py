import numpy as np
import matplotlib.pyplot as plt

# Modeling the challenge from Jet lag season
# Challenge: get heads 8 times in a row

class CoinFlipMarkov:
    def __init__(self, n, p = 0.5):
        self.states = np.arange(n + 1)  # States from 0 to n (0 heads to n heads)
        self.n = n
        self.p = p # probability of heads
        self.transMatrix = self.createP()

        if self.n < 1 or self.p < 0 or self.p > 1:
            raise ValueError("n must be >= 1 and/or p must be in [0, 1]")
        
    def createP(self):
        n = self.n
        p = self.p
        transMatrix = np.zeros((n + 1, n + 1), dtype=float)
        for i in range(n):
            transMatrix[i, i + 1] = p          # extend streak (go from 2 heads to 3 heads for example)
            transMatrix[i, 0]     = 1.0 - p    # tail resets streak to 0 (go from 2 heads to 0 heads if you get tails)
        transMatrix[n, n] = 1.0                # absorbing state (once you reach n heads, you stay there)
        return transMatrix
    
    def simulate(self, maxSteps, seed=None, return_path=False):
        rng = np.random.default_rng(seed)
        state = 0
        steps = 0
        path = [state]

        while state < self.n and steps < maxSteps:
            state = rng.choice(self.states, p=self.transMatrix[state])
            steps += 1
            if return_path:
                path.append(state)
        if return_path:
            return np.array(path)
        return state, steps

    def plotResults(self, bin_width=25):
        n = self.n
        results = self.simulate(1000, return_path=False)[1]  # Get the number of steps to reach n heads in a row
        data = np.asarray(results)
        lo, hi = data.min(), data.max()
        edges = np.arange(lo, hi + bin_width, bin_width)

        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=edges, density=True, alpha=0.7)
        plt.title(f'Distribution of Flips Needed to Reach {n} Heads in a Row')
        plt.xlabel('Number of Flips')
        plt.ylabel('Probability Density')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()



'''# test cases
if __name__ == "__main__":
    n = 8  # number of heads in a row
    p = 0.5  # probability of heads
    maxSteps = 10000  # maximum steps to simulate

    coin_flip = CoinFlipMarkov(n, p)
    # Simulate the coin flips
    # Plot the results
    coin_flip.plotResults(bin_width=15)'''


# Let's do the Jet Lag game for the steps
# Model: Start with an initial step rate of say, 1.84 steps/second
# Can you measure out say, 30 minutes using this rate (3312 steps)

class jetLagSteps:
    def __init__(self, stepRate, targetTime, transFreq, delta=0.01, minRate=0.5, maxRate=3.0, pSame=0.5, pUp=0.25, pDown=0.25):
        self.stepRate = stepRate        # e.g. 1.84
        self.targetTime = targetTime    # e.g. 30*60 seconds
        self.transFreq = transFreq      # e.g. 1 second per transition
        # discretization parameters
        self.delta = delta              # 0.01 steps/sec increments
        self.minRate = minRate      # minimum step rate (e.g. 0.5 steps/sec)
        self.maxRate = maxRate    # maximum step rate (e.g. 3.0 steps/sec)
        # transition probabilities
        self.pSame = pSame       # probability of staying in the same state
        self.pUp   = pUp        # probability of increasing step rate
        self.pDown = pDown      # probability of decreasing step rate

        # build state‐space and transition matrix
        self.states = np.arange(minRate, maxRate + delta, delta)
        self.P = self.createP()

    def createP(self):
        n = len(self.states)
        P = np.zeros((n, n))

        for i in range(n):
            # DOWN: to state i-1 (or reflect at boundary)
            if i > 0:
                P[i, i-1] = self.pDown
            else:
                P[i, i] += self.pDown

            # SAME: stay in i
            P[i, i] += self.pSame

            # UP: to state i+1 (or reflect at boundary)
            if i < n - 1:
                P[i, i+1] = self.pUp
            else:
                P[i, i] += self.pUp

        # Sanity check: each row should sum to 1
        assert np.allclose(P.sum(axis=1), 1), "Rows must sum to 1"
        return P

    import numpy as np

class JetLagSteps:
    def __init__(self, initRate,      # initial step‐rate (steps/sec), e.g. 1.84
        targetTime,    # total walking time (sec), e.g. 30*60
        transFreq,     # Δt between rate‐changes (sec), e.g. 1
        delta=0.01,     # discretization resolution (steps/sec)
        minRate=0.5,   # lowest rate
        maxRate=3.0,   # highest rate
        pSame=0.5,
        pUp=0.25,
        pDown=0.25
    ):
        self.initRate   = initRate
        self.targetTime = targetTime
        self.transFreq  = transFreq

        # discretize rates
        self.delta    = delta
        self.minRate = minRate
        self.maxRate = maxRate
        self.states   = np.arange(minRate, maxRate + delta/2, delta)

        # store transition probs
        self.pSame = pSame
        self.pUp   = pUp
        self.pDown = pDown

        # build P once
        self.P = self._buildTransitionMatrix()

    def _buildTransitionMatrix(self):
        n = len(self.states)
        P = np.zeros((n, n))

        for i in range(n):
            # step down
            if i > 0:
                P[i, i-1] = self.pDown
            else:
                # reflect at lower boundary
                P[i, i] += self.pDown

            # stay
            P[i, i] += self.pSame

            # step up
            if i < n - 1:
                P[i, i+1] = self.pUp
            else:
                # reflect at upper boundary
                P[i, i] += self.pUp

        # each row sums to 1
        assert np.allclose(P.sum(axis=1), 1), "Transition rows must sum to 1"
        return P

    def simulate(self, seed=None, returnPath=False):
        """
        Simulate random fluctuations in your step‐rate every transFreq seconds,
        walking until targetTime is reached.

        Returns:
          - if returnPath=False: (totalSteps, totalTime)
          - if returnPath=True:  an array of (time, rate, cumulativeSteps)
        """
        rng = np.random.default_rng(seed)

        rate          = self.initRate
        t             = 0.0
        cumulativeSt  = 0.0
        path          = [(t, rate, cumulativeSt)]

        while t < self.targetTime:
            # find current state index
            idx = int(round((rate - self.minRate) / self.delta))

            # sample next state
            idxNext = rng.choice(len(self.states), p=self.P[idx])
            rate = self.states[idxNext]

            # advance clock and steps
            t += self.transFreq
            cumulativeSt += rate * self.transFreq

            if returnPath:
                path.append((t, rate, cumulativeSt))

        if returnPath:
            return np.array(path, dtype=[('time', float),
                                         ('rate', float),
                                         ('steps', float)])
        return cumulativeSt, t
    




    
# create and print a transition matrix
if __name__ == "__main__":
    # Example usage
    stepRate = 1.84  # steps/sec
    targetTime = 10  # 30 minutes in seconds
    transFreq = 1  # change rate every second

    jetLag = JetLagSteps(stepRate, targetTime, transFreq)
    path = jetLag.simulate(returnPath=True)

   # print transition matrix P
    print("Transition Matrix P:")
    print(jetLag.P)
