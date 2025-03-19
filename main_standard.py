import sys
from traffic import TrafficGenerator
from trafficLightControlStandard import TrafficLightControlSimulation
from plot import Plot
from experiment import Experiment

"""
Main file to test a traffic simulation with reinforcement learning after training
"""

"""
numberOfInput: 4 (queueLength)
numberOfInput: 80 (positionVehicle)

python3 main_standard.py results_standard 1 4
python3 nameFile.py folderToSave numberOfExperiment numberOfInput
"""

if __name__ == "__main__":
    """
    Configuration(sumoGui, episodesTotal, maximumSteps, carsGenerated,
    greenLightDuration, yellowLightDuration, firstLayerWidth,
    secondLayerWidth, batchSize, learningRate, epochsTraining,
    minimumMemorySize, maximumMemorySize, statesInput, actionsOutput,
    gamma, pathModelOrResults, pathSumoConfiguration)
     """
    try:
        print("Start main_standard")

        Experiment_ = Experiment(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

        Configuration_ = Experiment_.getExperiment()

        numberExperiment = Experiment_.getNumberExperiment()

        TrafficGenerator_ = TrafficGenerator(Configuration_.getMaximumSteps(), Configuration_.getCarsGenerated())

        TrafficLightControlSimulation_ = TrafficLightControlSimulation(Configuration_, TrafficGenerator_)

        episode = 0

        while episode < Configuration_.getEpisodesTotal():
            print("************* Episode: " + str(episode + 1) + " *************")
            TrafficLightControlSimulation_.run(episode)
            episode += 1

        Plot(Configuration_.getPathModelOrResults()).printFile(
            'dataEpisodeStepAction_' + sys.argv[1] + '_' + str(numberExperiment) + '.csv',
            TrafficLightControlSimulation_.getStepActionInformation())

        Plot(Configuration_.getPathModelOrResults()).printFile(
            'dataCumulativeTime_' + sys.argv[1] + '_' + str(numberExperiment) + '.csv',
            TrafficLightControlSimulation_.getCumulativeWaitingTimeTotalEpisodes())

        print("Finish main_standard")

    except Exception as e:
        print(e)
        print('Exception: The arguments are not written correctly')