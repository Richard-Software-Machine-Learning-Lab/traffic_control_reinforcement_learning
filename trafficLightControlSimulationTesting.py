import numpy as np
from trafficLightControl import TrafficLightControl

"""
Traffic simulation with reinforcement learning while training with Q-learning
"""


class TrafficLightControlSimulation(TrafficLightControl):
    def __init__(self, Configuration, ModelTest, TrafficGenerator):
        super().__init__(Configuration, TrafficGenerator)
        self.ModelTest = ModelTest
        self.gamma_ = self.Configuration.getGamma()

        self.statesInput = Configuration.getStatesInput()
        self.epochsTraining = Configuration.getEpochsTraining()
        self.rewards = []
        self.cumulativeWaitingTime = []
        self.stepActionStateInformation = []

    def getModel(self):
        return self.ModelTest

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
            currentState = self.getStateInformation(self.Configuration.getStatesInput())

            print("The current state now  is: .....")
            print(currentState)

            currentTotalWaitingTime = self.getCollectiveWaitingTime()
            print("******* Current total waiting time *************")
            print(currentTotalWaitingTime)

            currentAction = self.getAction(currentState)

            print(currentAction)
            self.saveInfoPerStateTesting(episode, self.getStep(), currentAction, currentState)

            if self.getStep() != 0 and self.getPreviousAction() != currentAction:
                self.setYellowPhase(self.getPreviousAction())
                self.setStepsSimulation(self.yellowLightDuration)

            self.setGreenPhase(currentAction)
            self.setStepsSimulation(self.greenLightDuration)

            self.previousAction = currentAction
            self.previousTotalWaitingTime = currentTotalWaitingTime

            print("End while")
        self.saveInformationPerEpisode()

        """
        Ending Traci Simulation
        """

        print("Closing traci simulation ")
        self.setCloseTraci()

    def getStateInformation(self, stateInput):
        if stateInput == 4:
            return self.getStateLengthQueue()
        elif stateInput == 80:
            return self.getState()

    def saveInfoPerStateTesting(self, episode, step, currentAction, currentState):
        self.informationPerEpisodeStepActionReward = []
        self.informationPerEpisodeStepActionReward.append(episode)
        self.informationPerEpisodeStepActionReward.append(step)
        self.informationPerEpisodeStepActionReward.append(currentAction)
        self.informationStateEpisode.append(
            self.addCurrentState(self.informationPerEpisodeStepActionReward, currentState))

    def addCurrentState(self, informationPerEpisodeStepActionReward, currentState):
        self.informationWithElementsState = []
        self.informationWithElementsState.append(informationPerEpisodeStepActionReward[0])
        self.informationWithElementsState.append(informationPerEpisodeStepActionReward[1])
        self.informationWithElementsState.append(informationPerEpisodeStepActionReward[2])

        for stateElements in currentState:
            self.informationWithElementsState.append(stateElements)

        return self.informationWithElementsState

    def setInitialParametersEpisode(self):
        self.step_ = 0
        self.waitingTimes = {}
        self.sumNegativeRewards = 0
        self.sumWaitingTime = 0

        self.informationStateEpisode = []
        self.previousTotalWaitingTime = 0
        self.previousState = -1
        self.previousAction = -1

    def getPreviousTotalWaitingTime(self):
        return self.previousTotalWaitingTime

    def getPreviousState(self):
        return self.previousState

    def getWaitingTimes(self):
        return self.waitingTimes

    def getSumNegativeRewards(self):
        return self.sumNegativeRewards

    def getSumWaitingTime(self):
        return self.sumWaitingTime

    def getTraciStart(self):
        return self.startTraci

    def getAction(self, state):
        return self.ModelTest.getMaximumActions(self.ModelTest.getPredictionOneState(state))

    def getStateLengthQueue(self):
        state = np.zeros(self.statesInput)
        state[0] = self.getNumberOfVehiclesWithoutMovement("north_edge_one")
        state[1] = self.getNumberOfVehiclesWithoutMovement("east_edge_one")
        state[2] = self.getNumberOfVehiclesWithoutMovement("south_edge_one")
        state[3] = self.getNumberOfVehiclesWithoutMovement("west_edge_one")

        return state

    def getState(self):
        state = np.zeros(self.statesInput)
        """
        Returns a list of all objects in the network. e.g. ('E_W_11', 'N_S_12', 'W_E_10')
        """
        vehicleList = self.getVehiclesIdList()

        for vehicleIdentification in vehicleList:
            print("The vehicle identification ....")
            print(vehicleIdentification)
            """
            The position of the vehicle along the lane measured in m. e.g. 70.71
            """
            print("The position of the lane ....")
            positionLane = self.getLanePosition(vehicleIdentification)
            print(positionLane)
            """
            Returns the id if the lane the named vehicle was at within the last step. e.g. west_edge_one_1
            """
            identificationLane = self.getLaneId(vehicleIdentification)

            positionLane = 100 - positionLane
            """
            Returns the cell from the lane from 0 to 9
            """
            cellLane = self.getCellLane(positionLane)
            """
            Returns the lane identification in a clockwise manner from west to south
            """
            groupLane = self.getGroupLane(identificationLane)
            """
            Returns an array with the car's position and validity
            """
            positionValidCar = self.getPositionAndValidityCar(groupLane, cellLane)
            "Returns state with valid cars"
            state = self.getStateWithValidCars(state, positionValidCar)

        return state

    def getStateWithValidCars(self, state, positionValidCar):
        if positionValidCar[1]:
            state[positionValidCar[0]] = 1
        return state

    def getPositionAndValidityCar(self, groupLane, cellLane):
        positionValidCar = [0, False]
        if 1 <= groupLane <= 7:
            positionValidCar[0] = int(str(groupLane) + str(cellLane))
            positionValidCar[1] = True
        elif groupLane == 0:
            positionValidCar[0] = cellLane
            positionValidCar[1] = True
        else:
            positionValidCar[1] = False

        return positionValidCar

    def getGroupLane(self, identificationLane):
        if identificationLane == "west_edge_one_0":
            groupLane = 0
        elif identificationLane == "west_edge_one_1":
            groupLane = 1
        elif identificationLane == "north_edge_one_0":
            groupLane = 2
        elif identificationLane == "north_edge_one_1":
            groupLane = 3
        elif identificationLane == "east_edge_one_0":
            groupLane = 4
        elif identificationLane == "east_edge_one_1":
            groupLane = 5
        elif identificationLane == "south_edge_one_0":
            groupLane = 6
        elif identificationLane == "south_edge_one_1":
            groupLane = 7
        else:
            groupLane = -1

        return groupLane

    def getCellLane(self, positionLane):
        if positionLane < 10:
            cellLane = 0
        elif positionLane < 20:
            cellLane = 1
        elif positionLane < 30:
            cellLane = 2
        elif positionLane < 40:
            cellLane = 3
        elif positionLane < 50:
            cellLane = 4
        elif positionLane < 60:
            cellLane = 5
        elif positionLane < 70:
            cellLane = 6
        elif positionLane < 80:
            cellLane = 7
        elif positionLane < 90:
            cellLane = 8
        elif positionLane <= 100:
            cellLane = 9
        return cellLane

    def getLaneId(self, vehicleIdentification):
        identificationLane = self.getTraci().vehicle.getLaneID(vehicleIdentification)
        return identificationLane

    def getLanePosition(self, vehicleIdentification):
        positionLane = self.getTraci().vehicle.getLanePosition(vehicleIdentification)
        return positionLane

    def getVehiclesIdList(self):
        vehicleList = self.getTraci().vehicle.getIDList()
        return vehicleList

    def getCollectiveWaitingTime(self):
        waitingTimesDictionary = {}
        roadsWithTrafficLights = ["west_edge_one", "north_edge_one", "east_edge_one", "south_edge_one"]
        """
        Returns a list of all objects in the network. e.g. ('E_W_11', 'N_S_12', 'W_E_10')
        """
        vehicleList = self.getVehiclesIdList()
        for vehicleIdentification in vehicleList:
            """
            Returns the accumulated waiting time of a vehicle collects the vehicle's waiting time
            over a certain time interval (interval length is set per option '--waiting-time-memory')
            e.g. 35.0, 0.0, 51.0, waitingTimes: {'N_S_3': 0.0, 'E_W_11': 28.0, 'W_E_13': 11.0, 'W_S_14': 0.0}
            """
            waitingTime = self.getAccumulatedTimePerVehicleIdentification(vehicleIdentification)
            """
            Returns the id of the edge the named vehicle was at within the last step.
            e.g. west_edge_two, east_edge_two
            """
            roadIdentification = self.getRoadIdPerVehicleIdentification(vehicleIdentification)
            """
            e.g. waitingTimes: {'E_N_7': 0.0, 'E_W_1': 0.0, 'E_W_10': 0.0, 'E_W_15': 0.0, 'E_W_4': 0.0}
            """
            waitingTimesDictionary = self.getWaitingTimesDictionary(self.waitingTimes, roadsWithTrafficLights,
                                                                    vehicleIdentification, waitingTime,
                                                                    roadIdentification)

        totalWaitingTime = self.getTotalWaitingTime(waitingTimesDictionary.values())

        return totalWaitingTime

    def getWaitingTimesDictionary(self, waitingTimes, roadsWithTrafficLights, vehicleIdentification, waitingTime,
                                  roadIdentification):
        """
        Recognize the road and delete repeated vehicles with identification
        """
        if roadIdentification in roadsWithTrafficLights:
            waitingTimes[vehicleIdentification] = waitingTime

        return waitingTimes

    def getRoadIdPerVehicleIdentification(self, vehicleIdentification):
        roadIdentification = self.getTraci().vehicle.getRoadID(vehicleIdentification)
        return roadIdentification

    def getAccumulatedTimePerVehicleIdentification(self, vehicleIdentication):
        waitingTime = self.getTraci().vehicle.getAccumulatedWaitingTime(vehicleIdentication)
        return waitingTime

    def saveInformationPerEpisode(self):
        print("Total cumulative reward ...... ")
        print(self.sumNegativeRewards)
        self.rewards.append(self.sumNegativeRewards)
        self.cumulativeWaitingTime.append(self.sumWaitingTime)
        self.stepActionStateInformation.append(self.informationStateEpisode)

    def getStepActionStateInformation(self):
        return self.stepActionStateInformation

    def getRewardsListTotalEpisodes(self):
        return self.rewards

    def getCumulativeWaitingTimeTotalEpisodes(self):
        return self.cumulativeWaitingTime
