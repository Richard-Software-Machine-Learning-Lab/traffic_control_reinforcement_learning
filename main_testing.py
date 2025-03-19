import sys
from experiment import Experiment
from traffic import TrafficGenerator
from model import ModelTest
from trafficLightControlSimulationTesting import TrafficLightControlSimulation
from plot import Plot

"""
Main file to test a traffic simulation without reinforcement learning
"""

"""
python3 main_standard.py results_testing 1 4 30

python3 main_testing.py results_testing 1 4 40
python3 main_testing.py results_testing 3 4 160
python3 main_testing.py results_testing 3 4 320
python3 main_testing.py results_testing 3 4 400

python3 main_testing.py results_testing 2 80 80
python3 main_testing.py results_testing 2 80 160
python3 main_testing.py results_testing 2 80 320
python3 main_testing.py results_testing 2 80 400

python3 nameFile.py folderToSave numberOfExperiment numberOfInput numberEpisodeTraining
"""

if __name__ == "__main__":
    try:

        print("Start main_testing")

        Experiment_ = Experiment(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

        Configuration_ = Experiment_.getExperiment()

        numberExperiment = Experiment_.getNumberExperiment()

        episodeTraining = int(sys.argv[4])

        ModelTest_ = ModelTest(Configuration_.getPathModelOrResults(), Configuration_.getStatesInput(),
                               numberExperiment,
                               episodeTraining)

        TrafficGenerator_ = TrafficGenerator(Configuration_.getMaximumSteps(), Configuration_.getCarsGenerated())

        TrafficLightControlSimulation_ = TrafficLightControlSimulation(Configuration_, ModelTest_, TrafficGenerator_)

        episodeTesting = 0

        while episodeTesting < Configuration_.getEpisodesTotal():
            print("Episode: " + str(episodeTesting + 1))
            TrafficLightControlSimulation_.run(episodeTesting)
            episodeTesting += 1

        Plot(Configuration_.getPathModelOrResults()).printFile(
            'dataEpisodeStepActionRewardState_' + sys.argv[1] + '_' + str(numberExperiment) + '_' + str(episodeTraining) + '.csv',
            TrafficLightControlSimulation_.getStepActionStateInformation())

        Plot(Configuration_.getPathModelOrResults()).printFile(
            'dataRewards_' + sys.argv[1] + '_' + str(numberExperiment) + '_' + str(episodeTraining) + '.csv',
            TrafficLightControlSimulation_.getRewardsListTotalEpisodes())

        Plot(Configuration_.getPathModelOrResults()).printFile(
            'dataCumulativeTime_' + sys.argv[1] + '_' + str(numberExperiment) + '_' + str(episodeTraining) +  '.csv',
            TrafficLightControlSimulation_.getCumulativeWaitingTimeTotalEpisodes())

        print("Finish  main_testing")

    except Exception as e:
        print(e)
        print('Exception: The arguments are not written correctly')
