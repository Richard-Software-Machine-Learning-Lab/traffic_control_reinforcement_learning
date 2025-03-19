import unittest
from memory import Memory
from sample import Sample
from configuration import Configuration
from trafficLightControlSimulationTraining import TrafficLightControlSimulation
from model import ModelTrain
from traffic import TrafficGenerator
import numpy as np


class TestMemory(unittest.TestCase):
    def testSetSampleMemory(self):
        """
        Minimum memory size: 1
        Maximum memory size: 5
        """
        Memory_ = Memory(1, 5)
        """
        Sample
        """
        sample_ = Sample([], 0, -10, [])
        """
        Test Memory has no samples
        """
        self.assertEqual(Memory_.getSamplesSizeMemory(), 0)
        Memory_.setSample(sample_)
        """
        Test Memory has one sample
        """
        self.assertEqual(Memory_.getSamplesSizeMemory(), 1)
        Memory_.setSample(sample_)
        Memory_.setSample(sample_)
        """
        Test Memory the samples are more than the maximum setup
        """
        self.assertEqual(Memory_.getSamplesSizeMemory(), 3)

    def testMinimumSizeMemory(self):
        Memory_ = Memory(1, 7)
        self.assertEqual(Memory_.getMinimumSizeMemory(), 1)

    def testMaximumSizeMemory(self):
        Memory_ = Memory(1, 7)
        self.assertEqual(Memory_.getMaximumSizeMemory(), 7)

    def testSamples(self):
        Memory_ = Memory(1, 2)
        """
        Test samples when less than the minimum size memory setup
        """
        self.assertEqual(Memory_.getSamples(1), [])
        """
        Sample is a tuple
        (previousState, previousAction, reward, currentState)
        """
        sample_ = Sample([], 0, -10, [])

        Memory_.setSample(sample_)
        self.assertEqual(len(Memory_.getSamples(2)), 1)

        Memory_.setSample(sample_)
        self.assertEqual(len(Memory_.getSamples(2)), 2)


class TestConfiguration(unittest.TestCase):
    def testSumoGui(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, './', './')
        self.assertEqual(Configuration_.getSumoGui(), False)

    def testEpisodesTotal(self):
        Configuration_ = Configuration(False, 100, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, './', './')
        self.assertEqual(Configuration_.getEpisodesTotal(), 100)

    def testMaximumSteps(self):
        Configuration_ = Configuration(False, 0, 400, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, './', './')
        self.assertEqual(Configuration_.getMaximumSteps(), 400)

    def testCarsGenerated(self):
        Configuration_ = Configuration(False, 0, 0, 20, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, './', './')
        self.assertEqual(Configuration_.getCarsGenerated(), 20)

    def testGreenLightDuration(self):
        Configuration_ = Configuration(False, 0, 0, 0, 11, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, './', './')

        self.assertEqual(Configuration_.getGreenLightDuration(), 11)

    def testYellowLightDuration(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 4, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, './', './')

        self.assertEqual(Configuration_.getYellowLightDuration(), 4)

    def testFirstLayerWidth(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 0, 100, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, './', './')
        self.assertEqual(Configuration_.getFirstLayerWidth(), 100)

    def testSecondLayerWidth(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 0, 0, 90, 0,
                                       0, 0, 0, 0, 0, 0, 0, './', './')
        self.assertEqual(Configuration_.getSecondLayerWidth(), 90)

    def testBatchSize(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 0, 0, 0, 100,
                                       0, 0, 0, 0, 0, 0, 0, './', './')
        self.assertEqual(Configuration_.getBatchSize(), 100)

    def testLearningRate(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0.001, 0, 0, 0, 0, 0, 0, './', './')
        self.assertEqual(Configuration_.getLearningRate(), 0.001)

    def testEpochsTraining(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 600, 0, 0, 0, 0, 0, './', './')

        self.assertEqual(Configuration_.getEpochsTraining(), 600)

    def testMinimumMemorySize(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 20, 0, 0, 0, 0, './', './')

        self.assertEqual(Configuration_.getMinimumMemorySize(), 20)

    def testMaximumMemorySize(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 400, 0, 0, 0, './', './')
        self.assertEqual(Configuration_.getMaximumMemorySize(), 400)

    def testStatesInput(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 80, 0, 0, './', './')
        self.assertEqual(Configuration_.getStatesInput(), 80)

    def testActionsOutput(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 2, 0, './', './')
        self.assertEqual(Configuration_.getActionsOutput(), 2)

    def testGamma(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0.75, './', './')
        self.assertEqual(Configuration_.getGamma(), 0.75)

    def testPathModelNeuralNetwork(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, './models', './')
        self.assertEqual(Configuration_.getPathModelOrResults(), './models')

    def testPathModelNeuralNetwork(self):
        Configuration_ = Configuration(False, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, './', './environment/sumo_config.sumocfg')
        self.assertEqual(Configuration_.getPathSumoConfiguration(), './environment/sumo_config.sumocfg')


class TrafficSimulationTesting:
    def __init__(self):
        self.Configuration_ = Configuration(False, 300, 700, 30, 10, 4, 200, 200, 50, 0.001,
                                            800, 100, 50000, 80, 2, 0.75, './models', 'sumo_config.sumocfg')

        self.ModelTrain_ = ModelTrain(self.Configuration_.getFirstLayerWidth(),
                                      self.Configuration_.getSecondLayerWidth(),
                                      self.Configuration_.getBatchSize(),
                                      self.Configuration_.getLearningRate(), self.Configuration_.getStatesInput(),
                                      self.Configuration_.getActionsOutput())

        self.Memory_ = Memory(self.Configuration_.getMinimumMemorySize(), self.Configuration_.getMaximumMemorySize())

        self.TrafficGenerator_ = TrafficGenerator(self.Configuration_.getMaximumSteps(),
                                                  self.Configuration_.getCarsGenerated())

        self.TrafficLightControlSimulation_ = TrafficLightControlSimulation(self.Configuration_, self.ModelTrain_,
                                                                            self.Memory_,
                                                                            self.TrafficGenerator_)

    def getTrafficLightControlSimulation(self):
        return self.TrafficLightControlSimulation_


class TestTrafficLightControlSimulation(unittest.TestCase):
    def testTrafficLightControlSumoConfiguration(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        """
        Test sumo configuration
        """
        self.assertEqual(TrafficLightControlSimulation_.getSumoConfiguration(
            TrafficLightControlSimulation_.getConfiguration().getPathSumoConfiguration(),
            TrafficLightControlSimulation_.getConfiguration().getSumoGui(),
            TrafficLightControlSimulation_.getConfiguration().getMaximumSteps()),
            ['sumo', "-c", 'environment/sumo_config.sumocfg',
             "--no-step-log", "true", "--waiting-time-memory", '700'])

    def testTrafficLightControlMaximumSteps(self):
        """
        Test maximum steps in the simulation
        """
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        self.assertEqual(TrafficLightControlSimulation_.getMaximumSteps(), 700)

    def testTrafficLightControlSteps(self):
        """
         Test steps in the simulation
         """
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()

        self.assertEqual(TrafficLightControlSimulation_.getStep(), 0)

    def testTrafficLightActionsOutput(self):
        """
        Test number of actions output
        """
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        self.assertEqual(TrafficLightControlSimulation_.getActionsOutput(), 2)

    def testTrafficLightTraciStartAndStop(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        sumoConfiguration = TrafficLightControlSimulation_.getSumoConfiguration(
            TrafficLightControlSimulation_.getConfiguration().getPathSumoConfiguration(),
            TrafficLightControlSimulation_.getConfiguration().getSumoGui(),
            TrafficLightControlSimulation_.getConfiguration().getMaximumSteps())

        TrafficLightControlSimulation_.setRouteFileSimulation(1)
        TrafficLightControlSimulation_.setTraciStart(sumoConfiguration)
        self.assertEqual(TrafficLightControlSimulation_.getTraciStart(), True)
        TrafficLightControlSimulation_.setCloseTraci()
        self.assertEqual(TrafficLightControlSimulation_.getTraciStart(), False)

    def testTrafficLightInitialParametersEpisode(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        TrafficLightControlSimulation_.setInitialParametersEpisode()
        self.assertEqual(TrafficLightControlSimulation_.getStep(), 0)
        self.assertEqual(TrafficLightControlSimulation_.getWaitingTimes(), {})
        self.assertEqual(TrafficLightControlSimulation_.getSumNegativeRewards(), 0)

        self.assertEqual(TrafficLightControlSimulation_.getSumWaitingTime(), 0)
        self.assertEqual(TrafficLightControlSimulation_.getPreviousTotalWaitingTime(), 0)

        self.assertEqual(TrafficLightControlSimulation_.getPreviousState(), -1)
        self.assertEqual(TrafficLightControlSimulation_.getPreviousAction(), -1)

    def testTrafficLightVehicleIdList(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        sumoConfiguration = TrafficLightControlSimulation_.getSumoConfiguration(
            TrafficLightControlSimulation_.getConfiguration().getPathSumoConfiguration(),
            TrafficLightControlSimulation_.getConfiguration().getSumoGui(),
            TrafficLightControlSimulation_.getConfiguration().getMaximumSteps())

        TrafficLightControlSimulation_.setRouteFileSimulation(1)
        TrafficLightControlSimulation_.setTraciStart(sumoConfiguration)
        """
        Test vehicle id list
        """
        self.assertIsInstance(TrafficLightControlSimulation_.getVehiclesIdList(), tuple)
        TrafficLightControlSimulation_.setCloseTraci()

    def testTrafficLightCellLane(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        self.assertEqual(TrafficLightControlSimulation_.getCellLane(9), 0)
        self.assertEqual(TrafficLightControlSimulation_.getCellLane(12), 1)
        self.assertEqual(TrafficLightControlSimulation_.getCellLane(21), 2)
        self.assertEqual(TrafficLightControlSimulation_.getCellLane(31), 3)
        self.assertEqual(TrafficLightControlSimulation_.getCellLane(41), 4)
        self.assertEqual(TrafficLightControlSimulation_.getCellLane(51), 5)
        self.assertEqual(TrafficLightControlSimulation_.getCellLane(61), 6)
        self.assertEqual(TrafficLightControlSimulation_.getCellLane(71), 7)
        self.assertEqual(TrafficLightControlSimulation_.getCellLane(81), 8)
        self.assertEqual(TrafficLightControlSimulation_.getCellLane(91), 9)

    def testTrafficLightGroupLane(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        self.assertEqual(TrafficLightControlSimulation_.getGroupLane("west_edge_one_0"), 0)
        self.assertEqual(TrafficLightControlSimulation_.getGroupLane("west_edge_one_1"), 1)

        self.assertEqual(TrafficLightControlSimulation_.getGroupLane("north_edge_one_0"), 2)
        self.assertEqual(TrafficLightControlSimulation_.getGroupLane("north_edge_one_1"), 3)

        self.assertEqual(TrafficLightControlSimulation_.getGroupLane("east_edge_one_0"), 4)
        self.assertEqual(TrafficLightControlSimulation_.getGroupLane("east_edge_one_1"), 5)

        self.assertEqual(TrafficLightControlSimulation_.getGroupLane("south_edge_one_0"), 6)
        self.assertEqual(TrafficLightControlSimulation_.getGroupLane("south_edge_one_1"), 7)

        self.assertEqual(TrafficLightControlSimulation_.getGroupLane("na"), -1)

    def testTrafficLightPositionAndValidityCar(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        self.assertEqual(TrafficLightControlSimulation_.getPositionAndValidityCar(1, 2), [12, True])
        self.assertEqual(TrafficLightControlSimulation_.getPositionAndValidityCar(2, 5), [25, True])
        self.assertEqual(TrafficLightControlSimulation_.getPositionAndValidityCar(0, 5), [5, True])
        self.assertEqual(TrafficLightControlSimulation_.getPositionAndValidityCar(-1, 5), [0, False])

    def testTrafficLightPositionAndValidityCar(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        self.assertEqual(
            TrafficLightControlSimulation_.getStateWithValidCars(np.zeros(80, dtype=np.float32), [2, True])[2], 1)

    def testTrafficLightTotalWaitingTime(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        self.assertEqual(TrafficLightControlSimulation_.getTotalWaitingTime([2.0, 3.0, 5.0, 0.0, 7.0]), 17.0)

    def testTotalStepsSimulationWhenHigherThanMaximumSteps(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        self.assertEqual(TrafficLightControlSimulation_.getTotalStepsSimulationWhenHigherThanMaximumSteps(500, 400),
                         100)

    def testTotalStepsSimulationGivenMaximumSteps(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        self.assertEqual(TrafficLightControlSimulation_.getTotalStepsSimulationGivenMaximumSteps(498, 12, 500), 2)
        self.assertEqual(TrafficLightControlSimulation_.getTotalStepsSimulationGivenMaximumSteps(10, 12, 500), 12)

    def testTrafficLightNumberOfVehiclesWithoutMovement(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        sumoConfiguration = TrafficLightControlSimulation_.getSumoConfiguration(
            TrafficLightControlSimulation_.getConfiguration().getPathSumoConfiguration(),
            TrafficLightControlSimulation_.getConfiguration().getSumoGui(),
            TrafficLightControlSimulation_.getConfiguration().getMaximumSteps())

        TrafficLightControlSimulation_.setRouteFileSimulation(1)
        TrafficLightControlSimulation_.setTraciStart(sumoConfiguration)

        self.assertIsInstance(TrafficLightControlSimulation_.getNumberOfVehiclesWithoutMovement("north_edge_one"), int)

        TrafficLightControlSimulation_.setCloseTraci()

    def testTrafficLightTotalNumberOfVehiclesWithoutMovement(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        self.assertEqual(TrafficLightControlSimulation_.getTotalNumberOfVehiclesWithoutMovement(4, 1, 2, 0), 7)

    def testTrafficLightSumNegativeRewards(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        TrafficLightControlSimulation_.setInitialParametersEpisode()
        TrafficLightControlSimulation_.setSumNegativeRewards(-10)
        self.assertEqual(TrafficLightControlSimulation_.getSumNegativeRewards(), -10)

    def testTrafficLightSaveInformationPerEpisode(self):
        TrafficLightControlSimulation_ = TrafficSimulationTesting().getTrafficLightControlSimulation()
        TrafficLightControlSimulation_.setInitialParametersEpisode()
        TrafficLightControlSimulation_.setSumNegativeRewards(-10)
        TrafficLightControlSimulation_.setSumWaitingTime(22)
        TrafficLightControlSimulation_.saveInformationPerEpisode()
        self.assertEqual(TrafficLightControlSimulation_.getRewardsListTotalEpisodes(), [-10])
        self.assertEqual(TrafficLightControlSimulation_.getCumulativeWaitingTimeTotalEpisodes(), [22])


if __name__ == '__main__':
    unittest.main()
