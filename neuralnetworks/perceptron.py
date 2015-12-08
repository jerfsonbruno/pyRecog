def step(x):
	return 0 if x < 0.5 else 1

class Perceptron:

	def __init__(self, numberOfInputs, activationFunction=step):
		"""
		Constructor of class Perceptron.

		Parameters
		==========
		numberOfInputs : int 
		The number of attributes in your dataset
		* the class does NOT count *

		activationFunction : function
		The Perceptron's activation function.
		"""
		import random
		self.weights = []
		for i in range(0, numberOfInputs):
			self.weights += [random.random()]
		self.actvf = activationFunction

	def train(self, trainingSet, alpha=0.3, maximumEpochs=100):
		"""
		Trains the perceptron network to recognize the 
		classes in the training set.

		Parameters
		==========
		trainingSet : list<list>
		A python list matrix with the training set. Make sure that the class is 
		in the last column and it is numerical.
		
		alpha : float
		The learning rate with which the weights will be adjusted.

		maximumEpochs : int
		The maximum number of iterations for this training.
		"""
		import numpy
		stop = False
		epoch = 0
		while not stop and epoch < maximumEpochs:
			stop = True
			#print("\rTraining... "+str(epoch)),
			for instance in trainingSet:
				d = instance[-1]
				x = numpy.array([instance[:-1]])
				w = numpy.array([self.weights])
				u = numpy.dot(w,x.transpose())
				y = self.actvf(u)
				e = d - y;
				if e > 0:
					stop = False
				self.weights = self.weights + alpha * e * x
			epoch += 1

	def classify(self, pattern):
		"""
		Classifies the given pattern.

		Parameters
		==========
		pattern : list
		A Python list with a set of attributes. Must be same size 
		as number of inputs set when the instance was created.
		"""
		import numpy
		w = numpy.array([self.weights])
		x = numpy.array([pattern])
		u = numpy.dot(w,x.transpose())
		return self.actvf(u)




				
