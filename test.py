import neuralnetworks as nn
import matplotlib.pyplot as plt
import numpy as np


def testfn():
	nnet = nn.NeuralNetwork(1,[4,5],1)
	X = np.linspace(0, 10, 10).reshape((-1, 1))
	T = 1.5 + 0.6 * X + 0.8 * np.sin(1.5*X)
	T[np.logical_and(X > 2, X < 3)] *= 3
	T[np.logical_and(X > 5, X < 7)] *= 3

	nnet.train(X, T, nIterations = 1000)


	Y = nnet.use(X)
	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.plot(range(1, 11), T, color = 'g', linewidth = 1)
	ax.set(ylim = [0, 10], xlim = [0, 11])

	ax2 = fig.add_subplot(211)
	ax2.plot(range(1, 11), Y, color = 'r', linewidth = 1)
	ax2.set(ylim = [0, 10], xlim = [0, 11])
	plt.show()

	ax3 = fig.add_subplot(212)
	ax3.plot(range(1, 11), T, color = 'r', linewidth = 1)
	ax3.set(ylim = [0, 10], xlim = [0, 11])
	plt.show()

	ax4 = fig.add_subplot(212)
	ax4.plot(range(1, 11), Y, color = 'r', linewidth = 1)
	ax4.set(ylim = [0, 10], xlim = [0, 11])
	plt.show()

if __name__ == '__main__':
	X = np.linspace(0, 10, 10).reshape((-1, 1))
	T = 1.5 + 0.6 * X + 0.8 * np.sin(1.5*X)
	T[np.logical_and(X > 2, X < 3)] *= 3
	T[np.logical_and(X > 5, X < 7)] *= 3
	#testfn()

	A = np.array([1,2,3,4,5])
	B = np.array([1,2,3,4,7])
	print(B - A)
	