import math

import numpy as np


class TrafficGenerator:
    def __init__(self, maximumSteps, carsGenerated):
        self.maximumSteps_ = maximumSteps
        self.carsGenerated_ = carsGenerated

    def setRouteFileSimulation(self, seed):
        np.random.seed(seed)
        """
        Set a weibull distribution according to the number of vehicles generated
        """
        frequencyVehicles = np.random.weibull(1, self.carsGenerated_)
        """
        Order a weibull distribution according to the number of vehicles generated
        """
        frequencyVehicles = np.sort(frequencyVehicles)

        carSteps = []
        """
        Method rounds a number down to the nearest integer,
        """
        previousMinimum = math.floor(frequencyVehicles[1])
        """
        Method rounds a number up to the nearest integer
        """
        previousMaximum = math.ceil(frequencyVehicles[-1])

        newMinimum = 0
        newMaximum = self.maximumSteps_

        """
        Append values to the end of an array: carSteps
        """
        for position in frequencyVehicles:
            """
            Reshape the distribution to fit the interval from 0 to the maximum number of episodes
            """
            carStep = ((newMaximum - newMinimum) / (previousMaximum - previousMinimum)) * (
                    position - previousMaximum) + newMaximum
            carSteps = np.append(carSteps, carStep)

        """
        Round elements of the array to the nearest integer.
        """

        carSteps = np.rint(carSteps)

        with open("environment/episode_current_small_environment.rou.xml", "w") as movement:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />
            <route id="W_E" edges= "west_edge_one east_edge_two"/>
            <route id="N_S" edges= "north_edge_one south_edge_two"/>
            <route id="E_W" edges= "east_edge_one  west_edge_two"/>
            <route id="S_N" edges= "south_edge_one north_edge_two"/> 

            """, file=movement)

            for counter, step in enumerate(carSteps):
                """
                Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high)
                """
                carMovementDirectionStraight = np.random.uniform()
                # 20%
                if carMovementDirectionStraight < 0.20:
                    print(
                        '<vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" '
                        'departSpeed="10" />' % (
                            counter, step), file=movement)
                # 25%
                elif carMovementDirectionStraight < 0.45:
                    print(
                        '<vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" '
                        'departSpeed="10" />' % (
                            counter, step), file=movement)
                # 25%
                elif carMovementDirectionStraight < 0.70:
                    print(
                        '<vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" '
                        'departSpeed="10" />' % (
                            counter, step), file=movement)
                # 30%
                else:
                    print(
                        '<vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" '
                        'departSpeed="10" />' % (
                            counter, step), file=movement)

            print("</routes>", file=movement)

