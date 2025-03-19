import sys

import traci
import random
import os
from sumolib import checkBinary

"""
Traffic simulation without reinforcement learning
"""
NORTH_SOUTH_REVERSE_GREEN_PHASE = 0
EAST_WEST_REVERSE_GREEN_PHASE = 2


class TrafficLightControl:
    def __init__(self, Configuration, TrafficGenerator):
        self.Configuration = Configuration
        self.TrafficGenerator = TrafficGenerator
        self.traci = traci
        self.startTraci = False
        self.step_ = 0
        self.maximumSteps = Configuration.getMaximumSteps()
        self.greenLightDuration = Configuration.getGreenLightDuration()
        self.yellowLightDuration = Configuration.getYellowLightDuration()

        self.actionsOutput = Configuration.getActionsOutput()
        self.cumulativeWaitingTime = []
        self.stepActionInformation = []

    """
    Returns configuration for the TrafficLightControlSimulation
    """

    def getConfiguration(self):
        return self.Configuration

    """
    Returns traffic generator for the TrafficLightControlSimulation
    """

    def getTrafficGenerator(self):
        return self.TrafficGenerator

    """
    Returns the sumo configuration for the TrafficLightControlSimulation
    """

    def getSumoConfiguration(self, pathSumoConfiguration, sumoGui, maximumNumberSteps):
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit(" It is necessary to be declared the variable 'SUMO_HOME'")

        if not sumoGui:
            binarySumo = checkBinary('sumo')
        else:
            binarySumo = checkBinary('sumo-gui')

        sumoConfiguration = [binarySumo, "-c", os.path.join('environment', pathSumoConfiguration), "--no-step-log",
                             "true", "--waiting-time-memory", str(maximumNumberSteps)]

        return sumoConfiguration

    def setCloseTraci(self):
        self.getTraci().close()
        self.startTraci = False

    def setInitialParametersEpisode(self):
        self.step_ = 0
        self.waitingTimes = {}
        self.sumWaitingTime = 0
        self.previousAction = -1
        self.informationStateEpisode = []

    def getPreviousAction(self):
        return self.previousAction

    def getTraci(self):
        return self.traci

    def setTraciStart(self, sumoConfiguration):
        self.getTraci().start(sumoConfiguration)
        self.startTraci = True

    def setRouteFileSimulation(self, episode):
        self.TrafficGenerator.setRouteFileSimulation(episode)

    def saveInfoPerState(self, episode, step, currentAction):
        self.informationPerStateStep = []
        self.informationPerStateStep.append(episode)
        self.informationPerStateStep.append(step)
        self.informationPerStateStep.append(currentAction)
        self.informationStateEpisode.append(self.informationPerStateStep)

    def setStepsSimulation(self, stepsLightDuration):
        totalStepsSimulation = self.getTotalStepsSimulationGivenMaximumSteps(self.getStep(), stepsLightDuration,
                                                                             self.getMaximumSteps())
        while totalStepsSimulation > 0:
            self.setTraciSimulationStep()
            self.setStepPerEpisode(1)
            totalStepsSimulation -= 1
            self.setSumWaitingTime(self.getLengthQueue())

    def setSumWaitingTime(self, lengthQueue):
        self.sumWaitingTime += lengthQueue

    def setTraciSimulationStep(self):
        self.getTraci().simulationStep()

    def setStepPerEpisode(self, value):
        self.step_ += value

    def getTotalStepsSimulationGivenMaximumSteps(self, steps, stepsLightDuration, maximumSteps):
        totalStepsSimulation = steps + stepsLightDuration

        if totalStepsSimulation >= maximumSteps:
            totalStepsSimulation = self.getTotalStepsSimulationWhenHigherThanMaximumSteps(maximumSteps, steps)
        else:
            totalStepsSimulation = stepsLightDuration

        return totalStepsSimulation

    def getTotalStepsSimulationWhenHigherThanMaximumSteps(self, maximumSteps, step):
        totalStepsSimulation = maximumSteps - step
        return totalStepsSimulation

    def getLengthQueue(self):
        """
        Returns the total number of halting vehicles for the last time step on the given edge.
        A speed of less than 0.1 m/s is considered a halt. Number of vehicles without movement in a respective edge.
        """

        """
        Waiting time (number of vehicles) in the queue north
        """
        queueNorth = self.getNumberOfVehiclesWithoutMovement("north_edge_one")

        """
        Waiting time (number of vehicles) in the queue south
        """
        queueSouth = self.getNumberOfVehiclesWithoutMovement("east_edge_one")

        """
        Waiting time (number of vehicles) in the queue east
        """
        queueEast = self.getNumberOfVehiclesWithoutMovement("south_edge_one")

        """
        Waiting time (number of vehicles) in the queue west
        """
        queueWest = self.getNumberOfVehiclesWithoutMovement("west_edge_one")
        print("****** Queues ******* ")
        print(queueNorth)
        print(queueSouth)
        print(queueEast)
        print(queueWest)
        totalQueue = self.getTotalNumberOfVehiclesWithoutMovement(queueNorth, queueSouth, queueEast, queueWest)

        return totalQueue

    def getTotalNumberOfVehiclesWithoutMovement(self, queueNorth, queueSouth, queueEast, queueWest):
        totalQueue = queueNorth + queueSouth + queueEast + queueWest
        return totalQueue

    def getNumberOfVehiclesWithoutMovement(self, edge):
        queue = self.getTraci().edge.getLastStepHaltingNumber(edge)
        return queue

    def getAction(self, episode):
        """
        Returns a random integer N such that a <= N <= b.
        """
        return random.randint(0, self.getActionsOutput() - 1)

    """
    Set up the yellow color in the traffic light according to the .net.xml file
    """

    def setYellowPhase(self, previousAction):
        yellowPhasePositionTrafficLightId = previousAction + 1
        self.setPhaseLightId(yellowPhasePositionTrafficLightId)

    """
    Set up the green color in the traffic light according to the .net.xml file
    """

    def setGreenPhase(self, action):
        """
        Switches to the phase with the given index in the list of all phases for the current program.
        """
        if action == 0:
            self.setPhaseLightId(NORTH_SOUTH_REVERSE_GREEN_PHASE)
        elif action == 1:
            self.setPhaseLightId(EAST_WEST_REVERSE_GREEN_PHASE)

    """
    Set up movement traffic light ID phase
    """
    def setPhaseLightId(self, directionTrafficLightHeaderId):
        """
        setPhase(self, tlsID, index)
        """
        self.getTraci().trafficlight.setPhase("junction_center", directionTrafficLightHeaderId)

    def getActionsOutput(self):
        return self.actionsOutput

    def getStep(self):
        return self.step_

    def getMaximumSteps(self):
        return self.maximumSteps

    def getTotalWaitingTime(self, values):
        return sum(values)

    def saveInformationPerEpisode(self):
        self.cumulativeWaitingTime.append(self.sumWaitingTime)
        self.stepActionInformation.append(self.informationStateEpisode)

    def getStepActionInformation(self):
        return self.stepActionInformation

    def getCumulativeWaitingTimeTotalEpisodes(self):
        return self.cumulativeWaitingTime
