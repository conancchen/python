import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta


# Class for Poisson Process to simulate arrival times
class PoissonProcess:
    # constructor, Poisson only has attribute of lambda value (rate)
    def __init__(self, rate):
        self.rate = rate

    # returns rate of poisson proccess
    def getRate(self):
        return self.rate
    
    # Expecteation Functions
    # expected first arrival
    def expectedFirstArrival(self):
        return 1 / self.rate

    # expected nth arrival
    def expectedNthArrival(self, n):
        return n / self.rate
    
    # Simulation Functions
    # given time, simulate arrivals
    def simulateArrivalsCount(self, t):
        return np.random.poisson(self.rate * t)
        
    # simulate the first arrival
    def simulateFirstArrival(self):
        return np.random.exponential(1 / self.rate)

    # simulate n arrivals (returns array of times)
    def simulateNArrivals(self, n):
        arrivals = [self.simulateFirstArrival()]  # Initialize with the first arrival
        for i in range(1, n):
            arrivals.append(np.random.exponential(1 / self.rate) + arrivals[i-1])
        return arrivals
    
    # returns the nth arrival
    def simulateNthArrival(self, n, arrivals=None):
        if arrivals is None:
            arrivals = self.simulateNArrivals(n)
        return arrivals[n - 1]

    # simulate till time t
    def simulateUntilT(self, t):
        arrivals = []
        current_time = 0
        while True:
            current_time += np.random.exponential(1 / self.rate)
            if current_time > t:
                break
            arrivals.append(current_time)
        return arrivals


    # given array of proccess, combine
    def superposition(self, processes):
        additional = sum(process.rate for process in processes)
        superposed = PoissonProcess(self.rate + additional) #add rate to the new proccesses
        return superposed

    # separate a proccess based on probabilities
    def thinner(self, probabilities):
        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("Probabilities should sum to 1.")

        processes = [PoissonProcess(self.rate * p) for p in probabilities]
        return processes
    
    def thinSpecific(self, process):
        return process.rate / self.rate

# Create a BTS Line Object
class BTSLine:
    def __init__(self, line_name, start_time, direction):
        self.line_name = line_name
        self.start_time = datetime.strptime(start_time, "%H:%M")
        self.direction = direction # for Sukhunmivt "to Kheha" or "to Khu Khot"
        self.timetable = []  # list of tuples: (station, {line_name: [arrival_times]})

    def addStation(self, station):
        # Check if station already exists
        for i, (s, line_dict) in enumerate(self.timetable):
            if s == station:
                if self.line_name in line_dict:
                    raise ValueError(f"Line '{self.line_name}' already exists at station '{station}'")
                line_dict[self.line_name] = []
                return
        
        # If station is new, create it with this line
        self.timetable.append((station, {self.line_name: []}))

    # simulate arrivals for this line at all stations
    def simulateLine(self, interval, numTrains):
        rate = 1 / interval
        numStations = len(self.timetable)
        process = PoissonProcess(rate)
        stationOrder = self.timetable if self.direction == "to Kheha" else list(reversed(self.timetable))
        for _ in range(numTrains): # go through all each train
            # Generate arrival offsets (in minutes) for this train
            arrivalOffsets = process.simulateNArrivals(numStations)
            currentTimes = [self.start_time + timedelta(minutes=offset) for offset in arrivalOffsets]

            for (station, lineArrivals), arrivalTime in zip(stationOrder, currentTimes):
                # Avoid overlap with any other line at this station
                retries = 0
                while any(arrivalTime in arrivals for arrivals in lineArrivals.values()):
                    arrivalTime += timedelta(minutes=1)
                    retries += 1
                    if retries > 100:
                        raise RuntimeError(f"Too many overlaps trying to schedule train at '{station}'")

                if self.line_name not in lineArrivals:
                    lineArrivals[self.line_name] = []

                lineArrivals[self.line_name].append(arrivalTime)


    # simulates time to travel between two stations
    def simulateTravelTime(self, interval, stationStart, stationEnd, departureTime):
        # extract stations and count # of segments
        stationNames = [s for s, _ in self.timetable]
        indexStart = stationNames.index(stationStart)
        indexEnd = stationNames.index(stationEnd)
        n = abs(indexEnd - indexStart) 

        # Check for direction validity
        if self.direction == "to Kheha" and indexStart > indexEnd:
            raise ValueError("Invalid direction: this line travels 'to Kheha', but stationStart comes after stationEnd.")
        elif self.direction == "to Khu Khot" and indexStart < indexEnd:
            raise ValueError("Invalid direction: this line travels 'to Khu Khot', but stationStart comes before stationEnd.")

        process = PoissonProcess(1 / interval)
        travelTime = process.simulateNthArrival(n)
        arrivalTime = departureTime + timedelta(minutes=travelTime)

        return f"Arrival at: {arrivalTime.strftime('%H:%M')}, Travel Time: {travelTime} minutes"

    # simulates the segment between two stations, outputting arrival time at each intermediatry station
    def simulateSegment(self, interval, stationStart, stationEnd, departureTime):
        # Determine station list and direction
        stationNames = [s for s, _ in self.timetable]
        indexStart = stationNames.index(stationStart)
        indexEnd = stationNames.index(stationEnd)

        if indexStart == indexEnd:
            return f"{stationStart}: {departureTime.strftime('%H:%M:%S')}"

        direction = 1 if indexEnd > indexStart else -1
        stationRange = stationNames[indexStart:indexEnd + direction:direction]

        # Simulate Poisson arrival offsets
        process = PoissonProcess(1 / interval)
        segmentCount = len(stationRange) - 1
        travelOffsets = process.simulateNArrivals(segmentCount)

        # Convert offsets to arrival timestamps
        arrivals = [departureTime + timedelta(minutes=offset) for offset in travelOffsets]

        # Combine departure and arrivals into output
        outputLines = [f"{stationStart}: {departureTime.strftime('%H:%M:%S')}"]
        outputLines += [
            f"{station}: {arrival.strftime('%H:%M:%S')}"
            for station, arrival in zip(stationRange[1:], arrivals)
        ]

        return "\n".join(outputLines)

    def simAvgOfNTrips(self, interval, stationStart, stationEnd, departureTime, n):
        travelTimes = []
        for _ in range(n):
            sim = self.simulateTravelTime(interval, stationStart, stationEnd, departureTime)
            travelTimes.append(float(sim.split(", Travel Time: ")[1].split(" ")[0]))
        
        average_time = np.mean(travelTimes)  # Calculate the average travel time in minutes

        minutes = int(average_time // 1)
        seconds = int((average_time % 1) * 60)
        return f"Average Time ({minutes} min, {seconds} sec), Expected Arrival ({(departureTime + timedelta(minutes=average_time)).strftime('%H:%M')})"


    # Returns the total run time of the line from the first to last station
    def totalRunTime(self, lineName):
        if not self.timetable:
            return timedelta(0)

        try:
            startTime = self.timetable[0][1][lineName][0]
            endTime = self.timetable[-1][1][lineName][0]
            return endTime - startTime
        except (KeyError, IndexError):
            return timedelta(0)

    # Returns the time between two stations on a specific line
    def timeBetweenStations(self, stationStart, stationEnd, lineName):
        if stationStart == stationEnd:
            return "00:00"

        # Filter only stations where the specified line exists
        lineStations = [station for station, lineDict in self.timetable if lineName in lineDict]
        if stationStart not in lineStations or stationEnd not in lineStations:
            raise ValueError("station not valid")

        # Find arrival times for the first train at each station
        startTime = next(
            lineDict[lineName][0] for station, lineDict in self.timetable if station == stationStart
        )
        endTime = next(
            lineDict[lineName][0] for station, lineDict in self.timetable if station == stationEnd
        )

        # Compute time difference in seconds
        totalSeconds = abs((endTime - startTime).total_seconds())
        minutes = int(totalSeconds // 60)
        seconds = round(totalSeconds % 60)

        return f"{minutes:02d}:{seconds:02d}"

    # Returns the number of stops between two stations (number of "segments" or hops)
    def numberOfStopsBtwn(self, stationStart, stationEnd):
        if stationStart == stationEnd:
            return 0
        stationNames = [s for s, _ in self.timetable]
        try:
            indexStart = stationNames.index(stationStart)
            indexEnd = stationNames.index(stationEnd)
        except ValueError:
            raise ValueError("station not valid")

        return abs(indexEnd - indexStart)

    # Returns a flat list of tuples (station, line, time) for all arrivals
    def getTimetable(self):
        flat = []
        for station, line_dict in self.timetable:
            for line, times in line_dict.items():
                for t in times:
                    formatted = t.strftime("%H:%M")
                    flat.append((station, line, formatted))
        return sorted(flat, key=lambda x: x[2])  # sort by time

    def printTimetable(self):
        print(f"=== BTS {self.line_name} Line Timetable ===")
        for station, line, time in self.getTimetable():
            print(f"{line} train arrives at {station} at {time}")


# input stations for each BTS line
sukhumvitStations = [
    "Khu Khot", "Yaek Kor Por Aor", "Rangsit", "Yaek Rangsit", "Royal Thai Air Force Museum", "Bhumibol Adulyadej Hospital",
    "Sai Yud", "Saphan Mai", "Bhumibol Hospital", "Wong Sawang", "Phahonyothin 59", "Wat Phra Sri Mahathat",
    "11th Infantry Regiment", "Bang Bua", "Kasetsart University", "Sena Nikhom", "Ratchayothin", "Phahonyothin 24",
    "Ha Yaek Lat Phrao", "Mo Chit", "Saphan Khwai", "Ari", "Sanam Pao", "Victory Monument", "Phaya Thai",
    "Ratchathewi", "Siam", "Chit Lom", "Phloen Chit", "Nana", "Asok", "Phrom Phong",
    "Thong Lo", "Ekkamai", "Phra Khanong", "On Nut", "Bang Chak", "Punnawithi",
    "Udom Suk", "Bang Na", "Bearing", "Samrong", "Pu Chao", "Chang Erawan", "Royal Thai Naval Academy",
    "Pak Nam", "Sai Luat", "Kheha"
]

silomStations = ["National Stadium", "Siam", "Ratchadamri", "Sala Daeng", "Chong Nonsi", 
    "Saint Louis", "Surasak", "Saphan Taksin", "Krung Thon Buri", "Wongwian Yai", 
    "Pho Nimit", "Talat Phlu", "Wutthakat", "Bang Wa"
]

goldLineStations = ["Krung Thon Buri", "Charoen Nakhon", "Khlong San"]


# TEST CASES!!!!!
# === Setup: Initialize full Sukhumvit BTS line ===
bts = BTSLine("Sukhumvit", "8:00", "to Kheha")

# Add all stations
for station in sukhumvitStations:
    bts.addStation(station)

departure = datetime.strptime("16:30", "%H:%M")

'''# === TEST 1: timeBetweenStations ===
print("=== Test 1: Time Between Mo Chit and Phra Khanong ===")
# Note: simulateLine must be called first to generate arrivals
bts.simulateLine(interval=3, numTrains=1)
print(bts.timeBetweenStations("Mo Chit", "Phra Khanong", "Sukhumvit"))
# Expected Output Format: 23:00 (23 minutes, for example)


# === TEST 2: simulateSegment ===
print("\n=== Test 2: Simulate Segment from Mo Chit to Phra Khanong ===")
segment_output = bts.simulateSegment(interval=3, stationStart="Mo Chit", stationEnd="Phra Khanong", departureTime=departure)
print(segment_output)


# === TEST 3: simulateTravelTime ===
print("\n=== Test 3: Simulate Travel Time from Mo Chit to Phra Khanong ===")
travel_output = bts.simulateTravelTime(
    interval=3,
    stationStart="Mo Chit",
    stationEnd="Phra Khanong",
    departureTime=departure
)
print(travel_output)
# Expected Output Format:
# Arrival at: 06:35, Travel Time: 20.7 minutes


# === TEST 4: Print Timetable ===
print("\n=== Test 4: Full Timetable After Simulation ===")
bts.printTimetable()
# Expected Output Format:
# Sukhumvit train arrives at Khu Khot at 06:01
# Sukhumvit train arrives at Yaek Kor Por Aor at 06:03
# ...
# Sukhumvit train arrives at Kheha at 06:57
'''

# === TEST 5: Simulate N Journey Times from Phra Khanong to Phloen Chit ===
print("\n=== Test 5: Simulate 100 Journey Times from Phra Khanong to Phloen Chit ===")
commuteAvg = bts.simAvgOfNTrips(
    interval=2,
    stationStart="Phloen Chit",
    stationEnd="Phra Khanong",
    departureTime=departure,
    n=30
)
print(commuteAvg)