# Neural networks
import neuralnetworks as nn
import matplotlib.pyplot as plt
import numpy as np
import scaledconjugategradient as scg
import mlutils as ml

def trainNNs(X, T, trainFraction, hiddenLayerStructures, numberRepetitions, numberIterations, classify = False):
	# Master result list - we shall keep appending to this.
	result = []
	# Iterate through each network structure provided.
	for net in hiddenLayerStructures:
		# To store performances of each training run for a network structure.
		trainPerformance = []
		testPerformance = []
		# Iterate for number of repetitions to train neural network.
		for i in range(numberRepetitions):
			# Now, we have to partition X and T, into training and testing data.
			Xtrain,Ttrain,Xtest,Ttest = ml.partition(X, T, trainFraction, classification=classify)
			# Create a neural network for this structure.
			nnet = nnet = nn.NeuralNetwork(1,net,1)
			# Commence training
			nnet.train(Xtrain, Ttrain, nIterations = numberIterations)
			# Use the trained network to produce outputs (for both training and testing input datasets).
			trainOut = nnet.use(Xtrain)
			testOut = nnet.use(Xtest)
			# If classifying, calculate samples classified incorrectly (for both training and testing datasets).
			if classify == True:
				pass
			else:
				# Calculate error in training set.
				trainError = trainOut - Xtrain
				trainRMSE = np.sqrt(np.mean((trainError**2)))
				# Calculate error in testing set
				testError = testOut - Xtest
				testRMSE = np.sqrt(np.mean((testError**2)))

			# Append train and test performances to list.
			trainPerformance.append(trainRMSE)
			testPerformance.append(trainRMSE)

			# Now, we append everything to the master 'result' list.
			result.append([])


if __name__ == '__main__':
	hiddenLayerStructures = [[20], [5, 10, 20]]
	trainNNs(1, 1, 1, hiddenLayerStructures, 1, 1, 1)

