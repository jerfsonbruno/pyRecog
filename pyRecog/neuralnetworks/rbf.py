from ..distbased import kmeans, subx
from math import exp
from random import random
import numpy as np

class Gaussian:

	def __init__(self, center, variance):
		"""
		Constructor of the gaussian function.
		The objective here is for the user to specify the variance and center only 
		once.

		Parameters
		==========
		x : float
			The argument of the funcion.

		v : float
			The variance of the desired gaussian, or the standard deviation squared

		c : list
			The vector representing the center of the desired gaussian
		"""
		self.c = center
		self.v = variance
	
	def __call__(self, x):
		sub = 0
		if type(x) == float or type(x) == int:
			sub = (x-self.c)**2
		else:
			for i in range(len(x)):
				sub += (x[i] - self.c[i])**2
		if self.v == 0:
			g = 0
		else:
			g = -1 * ( float(sub)/float(2*self.v) )
		return exp(g)

class RBF:
	def __init__(self, nHiddenLayer, nOutputLayer, activationFunction=(lambda x: x)):
		"""
		Constructor of the RBF class.

		Parameters
		==========
		nHiddenLayer : int
			The number of neurons in the hidden layer of the network

		nOutputLayer : int
			The number of neurons in the output layer in the network
		"""
		self.nHiddenLayer = nHiddenLayer
		self.nOutputLayer = nOutputLayer
		self.rbfs = []
		self.weights = []
		self.actf = activationFunction

	def train(self, dataSet, errGoal, maxEpochs=100,learnRate=0.3,variance=False):
		"""
		Function for training the RBF model to fit the given training set.

		Parameters
		==========
		dataSet : list<list>
			A python list matrix with the training set. Make sure that the 
			class is in the last column and it is numerical.
		errGoal : float
			The maximum error allowed to complete the training.
		"""
		import sys
		# separate the class from the training set
		labels = []
		trainSet = []
		for instance in dataSet:
			trainSet += [instance[:-1]]
			labels += [instance[-1]]
		# train the hidden layer
		groups, centers = kmeans(self.nHiddenLayer, trainSet)
		variances = []
		if variance != False:
			variances = [variance] * len(centers)
		else:
			for ci in range(len(centers)):
				m = len(groups[ci])
				s = 0
				for instance_index in groups[ci]:
					instance = trainSet[instance_index]
					center = centers[ci]
					s2 = 0
					for i in range(len(center)):
						s2 += (instance[i] - center[i])**2
					s += s2
				variances += [s/float(m)] if m > 0 else [0] # just to make sure the result is a float
		# end of hidden layer training
		self.rbfs = []
		for i in range(len(centers)):
			self.rbfs += [Gaussian(centers[i], variances[i])]
		# train the output layer
		z = []
		for instance in trainSet:
			inputk = []
			for rbf in self.rbfs:
				inputk += [rbf(instance)]
			z += [inputk]
		nepochs = 0
		# initialize random weights
		self.weights = []
		for o in range(self.nOutputLayer):
			ws = []
			for a in range(len(z[0])+1): # +1 because of bias
				ws += [random() * 0.5] # *.5 to be small
			self.weights += [ws]
		self.weights = np.array(self.weights)
		for i in range(len(z)): #for each instance
			z[i] = [-1.0] + z[i] # add bias
		z = np.array(z)
		curr_error = 1.0
		while nepochs < maxEpochs and curr_error > errGoal:
			curr_error = 0.0
			for inst_index in range(len(z)):
				instance_error = 0.0
				for j in range(self.nOutputLayer):
					y3j = self.actf(np.dot(z[inst_index], self.weights[j].transpose()))
					deltinha = (labels[inst_index] - y3j)
					instance_error += (labels[inst_index] - y3j)**2
					for i in range(self.nHiddenLayer + 1): # +1 because of bias
						self.weights[j][i] = self.weights[j][i] + learnRate * deltinha * z[inst_index][i]
				instance_error = instance_error / 2.0
				curr_error += instance_error
			curr_error = curr_error/float(len(z))
			sys.stdout.write("\r"+str(curr_error)+" of mean squared error in epoch "+str(nepochs)+"/"+str(maxEpochs)+"                                                                     ")
			nepochs += 1
		return self.weights

	def __call__(self, instance):
		inputk = []
		for rbf in self.rbfs:
			inputk += [rbf(instance)]
		inputk = [-1.0] + inputk
		results = []
		for j in range(self.nOutputLayer):
			y3j = self.actf(np.dot(inputk, self.weights[j].transpose()))
			results += [y3j]
		return results

