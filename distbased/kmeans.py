from .distances import euclidean

def kmeans(k, datapoints):
	"""
	Implementation of the k-Means algorithm.

	Parameters
	==========
	k : int
		The number of centers you want to position

	datapoints : list<list>
		A python list with the datapoints
	"""
	centers = [-1] * k
	new_centers = datapoints[:k]
	center_sets = {}
	while centers != new_centers:
		centers = list(new_centers) # copy new centers to centers
		# reinitialize the mapping
		center_sets = {}
		for c in range(len(centers)):
			center_sets[c] = []
		# finished reinitializing the mapping
		# find the centers
		for dpoint_index in range(len(datapoints)):
			center = 0
			mind = euclidean(centers[0], datapoints[dpoint_index])
			for c in range(1,len(centers)):
				d = euclidean(centers[c], datapoints[dpoint_index])
				if d < mind:
					mind = d
					center = c
			center_sets[center] += [dpoint_index]
		# centers found
		# update centers
		for c in range(len(centers)):
			# for every center
			sigma = datapoints[center_sets[c][0]]
			ct = 1.0
			for dpoint_index in center_sets[c][1:]:
				sigma = sumx(sigma, datapoints[dpoint_index])
				ct += 1.0
			# update the center
			new_centers[c] = divx(sigma, ct)
	return center_sets

def sumx(xs,ys):
	if type(xs) != list:
		return xs + ys
	else:
		result = [0] * len(xs)
		for i in range(len(xs)):
			result[i] = sumx(xs[i],ys[i])
		return result

def divx(xs, y):
	if type(xs) != list:
		return xs / y
	else:
		return [x/y for x in xs]