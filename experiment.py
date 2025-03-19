from configuration import Configuration


class Experiment:
    def __init__(self, path, numberExperiment, numberInput):
        self.path = path
        self.numberExperiment = numberExperiment
        self.numberInput = numberInput

    def getExperiment(self):
        # Low Traffic: 500 cars
        # Experiment 1
        if self.getNumberExperiment() == 1:
            Configuration_ = Configuration(False, 10, 5000, 500, 11, 4, 200, 200, 100, 0.001,
                                           600, 500, 50000, self.numberInput, 2, 0.50, self.getPathExperiment(), 'sumo_config.sumocfg')
        # Experiment 2
        elif self.getNumberExperiment() == 2:
            Configuration_ = Configuration(False, 400, 5000, 500, 11, 4, 200, 200, 100, 0.001,
                                           600, 500, 50000,  self.numberInput, 2, 0.75, self.getPathExperiment(), 'sumo_config.sumocfg')

        # High Traffic: 3000 cars
        # Experiment 3
        elif self.getNumberExperiment() == 3:
            Configuration_ = Configuration(False, 400, 5000, 3000, 11, 4, 200, 200, 100, 0.001,
                                           600, 500, 50000,  self.numberInput, 2, 0.50, self.getPathExperiment(), 'sumo_config.sumocfg')
        # Experiment 4
        elif self.getNumberExperiment() == 4:
            Configuration_ = Configuration(False, 400, 5000, 3000, 11, 4, 200, 200, 100, 0.001,
                                           600, 500, 50000,  self.numberInput, 2, 0.75, self.getPathExperiment(), 'sumo_config.sumocfg')

        return Configuration_

    def getPathExperiment(self):
        return self.path

    def getNumberExperiment(self):
        return self.numberExperiment
