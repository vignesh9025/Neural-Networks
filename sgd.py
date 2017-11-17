import numpy as np
import matplotlib.pyplot as plt
import pdb 
import time

# Neural network code.
def neural():
	# Create training dataset - input data X and target values T.
	X = np.random.uniform(1, 10, 1000)
	T = np.random.uniform(1, 2, 1000)
	eta = 0.01
	# Initialize weights. 
	a = 0.0
	b = 0.0
	pred = []
	errGrad = np.array([0,0])
	# Initialize an empty numpy array to store predicted values.
	#predicted = np.array([])
	wsa = []
	wsb = []
	# Start loop. Looping through the input values 2000 times.
	for step in range(2000):
		step += 1 # To get step to go from 1 to 2000, instead of 0 to 1999.
		n = step % 1000
		# Compute the predicted value for current input. Use this to update weights and carry on. 
		predicted = a + b*X[n]
		pred.append(predicted)
		# Now calculate the gradient of the squared error w.r.t each weight.
		# Calculate the difference between the predicted and targer values first.
		err = T[n] - predicted
		# Compute error gradient w.r.t each weight.
		# Error gradient w.r.t w[0]
		errGrad[0] = - err*1
		# Error gradient w.r.t w[1]
		errGrad[1] = - err * X[n]
		# Update weights with this gradient and eta.
		a -= eta * errGrad[0]
		b -= eta * errGrad[1]
		wsa.append(a)
		wsb.append(b)

	# Plotting
	fig = plt.figure(figsize = (20, 5))

	# Plot 1 - predicted values.
	ax = fig.add_subplot(2, 1, 1)
	ax.plot(range(1, 2001), pred, linewidth = 1, color = 'g')
	ax.set(ylim = [0, 10], xlim = [1900, 1950])

	# Plot 2 - target values.
	F = np.concatenate((T, T))
	ax2 = fig.add_subplot(2, 1, 1)
	ax2.plot(range(1, 2001), F.flatten(), linewidth = 1, color = 'r')
	ax2.set(ylim = [0, 10], xlim = [1900, 1950])

	# Plot 3 - w0
	ax3 = fig.add_subplot(2, 1, 2)
	ax3.plot(range(1, 2001), wsa, linewidth = 1, color = 'r')
	ax3.set(ylim = [0, 10])

	# Plot 4 - w1
	ax4 = fig.add_subplot(2, 1, 2)
	ax4.plot(range(1, 2001), wsb, linewidth = 1, color = 'g')
	ax4.set(ylim = [0, 10])


	plt.show()


if __name__ == '__main__':
	neural()