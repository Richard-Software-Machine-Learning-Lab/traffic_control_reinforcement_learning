
class Configuration:
    def __init__(self, sumoGui, episodesTotal, maximumSteps, carsGenerated,
                 greenLightDuration, yellowLightDuration, firstLayerWidth,
                 secondLayerWidth, batchSize, learningRate, epochsTraining,
                 minimumMemorySize, maximumMemorySize, statesInput, actionsOutput,
                 gamma, pathModelOrResults, pathSumoConfiguration):
        self.sumoGui_ = sumoGui
        self.episodesTotal_ = episodesTotal
        self.maximumSteps_ = maximumSteps
        self.carsGenerated_ = carsGenerated
        self.greenLightDuration_ = greenLightDuration
        self.yellowLightDuration_ = yellowLightDuration
        self.firstLayerWidth_ = firstLayerWidth
        self.secondLayerWidth_ = secondLayerWidth
        self.batchSize_ = batchSize
        self.learningRate_ = learningRate
        self.epochsTraining_ = epochsTraining
        self.minimumMemorySize_ = minimumMemorySize
        self.maximumMemorySize_ = maximumMemorySize
        self.statesInput_ = statesInput
        self.actionsOutput_ = actionsOutput
        self.gamma_ = gamma
        self.pathModelOrResults = pathModelOrResults
        self.pathSumoConfiguration_ = pathSumoConfiguration

    def getSumoGui(self):
        return self.sumoGui_

    def getEpisodesTotal(self):
        return self.episodesTotal_

    def getMaximumSteps(self):
        return self.maximumSteps_

    def getCarsGenerated(self):
        return self.carsGenerated_

    def getGreenLightDuration(self):
        return self.greenLightDuration_

    def getYellowLightDuration(self):
        return self.yellowLightDuration_

    def getFirstLayerWidth(self):
        return self.firstLayerWidth_

    def getSecondLayerWidth(self):
        return self.secondLayerWidth_

    def getBatchSize(self):
        return self.batchSize_

    def getLearningRate(self):
        return self.learningRate_

    def getEpochsTraining(self):
        return self.epochsTraining_

    def getMinimumMemorySize(self):
        return self.minimumMemorySize_

    def getMaximumMemorySize(self):
        return self.maximumMemorySize_

    def getStatesInput(self):
        return self.statesInput_

    def getActionsOutput(self):
        return self.actionsOutput_

    def getGamma(self):
        return self.gamma_

    def getPathModelOrResults(self):
        return self.pathModelOrResults

    def getPathSumoConfiguration(self):
        return self.pathSumoConfiguration_
