import numpy as np
import matplotlib.pyplot as plt

# Python Simulation of the classic Drunk Passanger Problem
# If Passanger #1 on an airline is drunk, he will randomly choose a seat.
# You are passanger n. All other passangers will sit in their assigned seats unless it is taken
# in that case they will randomly choose a seat
# what is the chance you get your seat? Expected Ans: 1/2

class DrunkPassenger:
    def __init__(self, passengers):
        self.passengers = passengers        
        # create empty array of size passengers to store the seat info
        self.seats = np.zeros(passengers, dtype=int)

    def simulate(self):
        self.seats = np.zeros(self.passengers, dtype=int)
        
        # Drunk passenger (#1) chooses a random seat
        chosenSeat = np.random.randint(0, self.passengers)
        self.seats[chosenSeat] = 1  # passenger #1 is assigned number 1

        # Remaining passengers (#2 to #n)
        for i in range(1, self.passengers):
            if self.seats[i] == 0:
                self.seats[i] = i + 1  # passenger number is i+1
            else:
                emptySeats = np.where(self.seats == 0)[0]
                if len(emptySeats) > 0:
                    seat = np.random.choice(emptySeats)
                    self.seats[seat] = i + 1

    def getSeats(self):
        # make sure simulation has been run. if not, run the sim
        if not hasattr(self, 'seats') or self.seats is None or len(self.seats) == 0:
            self.simulate()
        return self.seats

    def correctSeat(self, passengerNumber=None):
        # Default to last passenger (number N)
        if passengerNumber is None:
            passengerNumber = self.passengers
        idx = passengerNumber - 1  # convert to 0-based index
        return self.seats[idx] == passengerNumber

    def simulateN(self, iterations):
        count = 0
        for _ in range(iterations):
            self.simulate()
            if self.correctSeat():  # checks passenger N
                count += 1
        proportion = round(count / iterations, 3)
        return f"You get the correct seat about {proportion} of the time."

# run test cases
if __name__ == "__main__":
    dp = DrunkPassenger(100)
    # dp.simulate()
    # print("Seats after simulation:", dp.getSeats())
    # print("Is passenger 100 in their correct seat?", dp.correctSeat(100))
    # print(dp.simulateN(100000)) # should be around 0.5



# now we want to see what happens if there are multiple drunk passangers

class DrunkPassengers:
    def __init__(self, passengers, drunkPassengers):
        self.passengers = passengers
        self.drunkPassengers = drunkPassengers  # e.g. [1, 4, 7]
        self.seats = np.zeros(passengers, dtype=int)

    def simulate(self):
        # reset all seats to empty (0)
        self.seats = np.zeros(self.passengers, dtype=int)
        # each passenger in order 1 to n takes a seat
        for i in range(self.passengers):
            pn = i + 1  # convert from 0 index
            # if this passenger is sober AND their own seat i is free, they will take it
            if pn not in self.drunkPassengers and self.seats[i] == 0:
                self.seats[i] = pn
            # otherwise they are effectively drunk: pick a random free seat
            else:
                empty = np.where(self.seats == 0)[0]
                if len(empty) > 0:
                    seatChoice = np.random.choice(empty)
                    self.seats[seatChoice] = pn
    
    def getSeats(self):
        # run sim if not done yet
        if np.count_nonzero(self.seats) == 0:
            self.simulate()
        return self.seats
    
    def correctSeat(self, passengerNumber=None):
        # default to last passenger if not specified
        if passengerNumber is None:
            passengerNumber = self.passengers
        return self.seats[passengerNumber - 1] == passengerNumber

    def simulateN(self, iterations):
        count = 0
        for _ in range(iterations):
            self.simulate()
            # measure for passenger N
            if self.correctSeat():
                count += 1
        prop = round(count / iterations, 3)
        return f"You get the correct seat about {prop} of the time."


dp = DrunkPassengers(1000, [1, 5, 20, 25, 30, 35, 60, 88])  # passengers 1,5,20 are “drunk”
dp.simulate()
print("Passenger 1000 correct?", dp.correctSeat(1000))
print(dp.simulateN(1000))

# notice how as k passengers are drunk the probability passanger n gets their seat is 1 / (n+1)... wow

