import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


class ModelTrain(nn.Module):
    def __init__(self, inputLayerOne, inputLayerTwo, batchSize,
                 learningRate, inputState, outputActions):
        super(ModelTrain, self).__init__()
        self.inputLayerOne_ = inputLayerOne
        self.inputLayerTwo_ = inputLayerTwo
        self.batchSize_ = batchSize
        self.learningRate_ = learningRate
        self.inputState_ = inputState
        self.outputActions_ = outputActions

        # Linear function
        self.linearOne_ = nn.Linear(self.inputState_, self.inputLayerOne_)
        self.linearTwo_ = nn.Linear(self.inputLayerOne_, self.inputLayerTwo_)
        self.linearThree_ = nn.Linear(self.inputLayerTwo_, self.outputActions_)

        # Optimizer
        self.optimizer_ = optim.Adam(self.parameters(), lr=self.learningRate_)

        # Criterion
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def getForwardOutput(self, state):
        inputLayer = F.relu(self.getFirstLinearFunction(state))
        inputLayer = F.relu(self.getSecondLinearFunction(inputLayer))
        output = self.getThirdLinearFunction(inputLayer)
        return output

    def getFirstLinearFunction(self, state):
        return self.linearOne_(state)

    def getSecondLinearFunction(self, inputLayer):
        return self.linearTwo_(inputLayer)

    def getThirdLinearFunction(self, inputLayer):
        return self.linearThree_(inputLayer)

    def getPredictionOneState(self, state):
        newState = torch.from_numpy(state.astype(np.float32))
        newState = torch.reshape(newState, (-1, self.inputState_)).to(self.device)
        return self.getForwardOutput(newState.requires_grad_())

    def getOutputActions(self):
        return self.outputActions_

    def getPredictionBatch(self, states):
        predictions = np.zeros((len(states), self.getOutputActions()))

        for counter, state in enumerate(states):
            predictions[counter] = self.getPredictionOneState(state).cpu().detach().numpy()

        return predictions

    def getTrainBatch(self, states, qTarget):
        # Clear out gradients
        self.optimizer_.zero_grad()

        newStates = torch.from_numpy(states.astype(np.float32))
        newStates = torch.reshape(newStates, (-1, self.inputState_)).to(self.device)
        qPredictedTorch = self.getForwardOutput(newStates).to(self.device)
        print("qPredicted Torch")

        qTargetTorch = torch.from_numpy(qTarget.astype(np.float32))
        qTargetTorch = torch.reshape(qTargetTorch, (-1, self.getOutputActions())).to(self.device)
        print("qTarget Torch")

        loss = self.loss(qTargetTorch, qPredictedTorch).to(self.device)
        # Backward propagation
        loss.backward()
        # Update the weights
        self.optimizer_.step()

    def getBatchSize(self):
        return self.batchSize_

    def getMaximumActions(self, actions):
        return np.argmax(actions.cpu().detach().numpy())

    def getSaveModel(self, model, path, numberExperiment, episode):
        torch.save(model, path + '/trainedModel_' + str(numberExperiment) + '_' + str(episode) + '.pt')


class ModelTest:
    def __init__(self, modelPath, inputState, numberExperiment, episode):
        self.model = self.getLoadingModel(modelPath, numberExperiment, episode)
        self.inputState_ = inputState
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def getLoadingModel(self, modelFolderPath, numberExperiment, episode):
        print(modelFolderPath + '/trainedModel_' + str(numberExperiment) + '_' + str(episode) + '.pt')
        return torch.load(modelFolderPath + '/trainedModel_' + str(numberExperiment) + '_' + str(episode) + '.pt', map_location=torch.device('cpu') )

    def getPredictionOneState(self, state):
        newState = torch.from_numpy(state.astype(np.float32))
        newState = torch.reshape(newState, (-1, self.inputState_)).to(self.device)
        return  self.model.getForwardOutput(newState.requires_grad_())

    def getMaximumActions(self, actions):
        return np.argmax(actions.cpu().detach().numpy())
