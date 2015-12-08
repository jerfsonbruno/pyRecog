from .distances import euclidean
import numpy

class KNN:

	def __init__(self, k, trainingset, distanceFunction=euclidean):
		self.k = k
		self.distf = distanceFunction
		neighbors = []
		clss = []
		for line in trainingset:
			neighbors += [line[:-1]]
			clss += [line[-1]]
		self.neighbors = numpy.array(neighbors)
		self.clss = numpy.array(clss)

	def classify(self, datapoint):
		from scipy import stats
		import copy
		dist = map(self.distf, repeat(datapoint,len(self.neighbors)), \
		 self.neighbors)
		ds = copy.copy(dist)
		dist.sort()
		clss = []
		for i in range(self.k):
			ind = ds.index(dist[i])
			clss += [self.clss[ind]]
		return int(stats.mode(clss)[0][0])

def repeat(array, times):
	r = []
	for i in range(0, times):
		r += [array]
	return r
