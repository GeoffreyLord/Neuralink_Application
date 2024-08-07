
class DrgnflyObj:
    def __init__(self, trainingDataMax, trainingDataMin, inputFunct, trainingDataSize, testingDataSize,
                 numInputNodes, numNodesPerHiddenLayer, numOutputNodes, numEpochs, learningRate, accuracyThreshold, 
                 rateChangeBool, loadWeights, numHiddenLayers):
        
        #Model Configuration
        self.trainingDataMax: trainingDataMax
        self.trainingDataMin: trainingDataMin
        self.inputFunct: inputFunct
        self.trainingDataSize: trainingDataSize
        self.testingDataSize: testingDataSize
        self.numInputNodes: numInputNodes
        self.numNodesPerHiddenLayer: numNodesPerHiddenLayer
        self.numOutputNodes: numOutputNodes
        self.numEpochs: numEpochs #This Should Eventually be determiend by the script
        self.learningRate: learningRate #This Should Eventually be determiend by the script
        self.accuracyThreshold: accuracyThreshold 
        self.rateChangeBool: rateChangeBool #this should eventually be removed when the learning rate is dynamic
        self.loadWeights: loadWeights #This should be true if parameters are provided. 
        self.numHiddenLayers: numHiddenLayers

        #Model Parameters
