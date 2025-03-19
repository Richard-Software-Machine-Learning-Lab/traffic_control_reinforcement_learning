import sys

import traci
import random
import os
from sumolib import checkBinary
from trafficLightControl import TrafficLightControl

"""
Traffic simulation without reinforcement learning
"""
NORTH_SOUTH_REVERSE_GREEN_PHASE = 0
EAST_WEST_REVERSE_GREEN_PHASE = 2


class TrafficLightControlSimulation(TrafficLightControl):
    def __init__(self, Configuration, TrafficGenerator):
        super().__init__(Configuration, TrafficGenerator)

    """
    Run the traffic light simulation
    """

    def run(self, episode):
        sumoConfiguration = self.getSumoConfiguration(self.Configuration.getPathSumoConfiguration(),
                                                      self.Configuration.getSumoGui(),
                                                      self.Configuration.getMaximumSteps())
        self.setRouteFileSimulation(episode)
        """
        Starting Traci simulation
        """
        print("Starting traci simulation ")
        self.setTraciStart(sumoConfiguration)
        self.setInitialParametersEpisode()
        while self.getStep() < self.getMaximumSteps():
            print("Beginning while ")

            currentAction = self.getAction(self.getStep())

            self.saveInfoPerState(episode, self.getStep(), currentAction)

            if self.getStep() != 0 and self.getPreviousAction() != currentAction:
                self.setYellowPhase(self.getPreviousAction())
                self.setStepsSimulation(self.yellowLightDuration)

            self.setGreenPhase(currentAction)
            self.setStepsSimulation(self.greenLightDuration)

            self.previousAction = currentAction

            print("End while")
        self.saveInformationPerEpisode()

        """
        Ending Traci Simulation
        """
        print("Closing traci simulation ")
        self.setCloseTraci()
