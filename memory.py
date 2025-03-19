import random


class Memory:
    def __init__(self, minimumSize, maximumSize):
        self.minimumSize_ = minimumSize
        self.maximumSize_ = maximumSize
        self.samples_ = []

    def getMinimumSizeMemory(self):
        return self.minimumSize_

    def getMaximumSizeMemory(self):
        return self.maximumSize_

    def getSamplesSizeMemory(self):
        return len(self.samples_)

    def getAllSamples(self):
        return self.samples_

    def getSamples(self, numberOfSamples):
        if self.getSamplesSizeMemory() < self.getMinimumSizeMemory():
            return []
        if numberOfSamples > self.getSamplesSizeMemory():
            return random.sample(self.getAllSamples(), self.getSamplesSizeMemory())
        else:
            return random.sample(self.getAllSamples(), numberOfSamples)

    def setSample(self, sample):
        self.samples_.append(sample)
