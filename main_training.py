import sys
from experiment import Experiment
from traffic import TrafficGenerator
from model import ModelTrain
from memory import Memory
from trafficLightControlSimulationTraining import TrafficLightControlSimulation
from plot import Plot

"""
Main file to test a traffic simulation without reinforcement learning
"""

"""
python3 main_training.py results_training 1 4
python3 nameFile.py folder experimentNumber inputNumber
"""

if __name__ == "__main__":
    try:
        print("Start main_training")

        Experiment_ = Experiment('./' + sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

        Configuration_ = Experiment_.getExperiment()

        numberExperiment = Experiment_.getNumberExperiment()

        ModelTrain_ = ModelTrain(Configuration_.getFirstLayerWidth(), Configuration_.getSecondLayerWidth(),
                                 Configuration_.getBatchSize(),
                                 Configuration_.getLearningRate(), Configuration_.getStatesInput(),
                                 Configuration_.getActionsOutput())

        Memory_ = Memory(Configuration_.getMinimumMemorySize(), Configuration_.getMaximumMemorySize())

        TrafficGenerator_ = TrafficGenerator(Configuration_.getMaximumSteps(), Configuration_.getCarsGenerated())

        TrafficLightControlSimulation_ = TrafficLightControlSimulation(Configuration_, ModelTrain_, Memory_,
                                                                       TrafficGenerator_)

        episode = 0

        while episode < Configuration_.getEpisodesTotal():
            print("Episode: " + str(episode + 1))
            epsilon = 1.0 - (episode / Configuration_.getEpisodesTotal())
            TrafficLightControlSimulation_.run(episode, epsilon)
            if (episode + 1) % 40 == 0:
                ModelTrain_.getSaveModel(ModelTrain_, Configuration_.getPathModelOrResults(), numberExperiment,
                                         episode + 1)

            episode += 1

        Plot(Configuration_.getPathModelOrResults()).printFile(
            'dataEpisodeStepActionRewardState_' + sys.argv[1] + '_' + str(numberExperiment) + '.csv',
            TrafficLightControlSimulation_.getStepActionStateInformation())

        Plot(Configuration_.getPathModelOrResults()).printFile(
            'dataRewards_' + sys.argv[1] + '_' + str(numberExperiment) + '.csv',
            TrafficLightControlSimulation_.getRewardsListTotalEpisodes())

        Plot(Configuration_.getPathModelOrResults()).printFile(
            'dataCumulativeTime_' + sys.argv[1] + '_' + str(numberExperiment) + '.csv',
            TrafficLightControlSimulation_.getCumulativeWaitingTimeTotalEpisodes())

        print("Finish main_training")

    except Exception as e:
        print(e)
        print('Exception: The arguments are not written correctly')
