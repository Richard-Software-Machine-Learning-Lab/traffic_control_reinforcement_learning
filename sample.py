
class Sample:
    def __init__(self, previousState, previousAction, reward, currentState):
        self.previousState = previousState
        self.previousAction = previousAction
        self.reward = reward
        self.currentState = currentState

    def getPreviousState(self):
        return self.previousState

    def getPreviousAction(self):
        return self.previousAction

    def getReward(self):
        return self.reward

    def getCurrentState(self):
        return self.currentState



