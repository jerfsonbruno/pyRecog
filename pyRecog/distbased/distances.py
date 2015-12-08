
def manhatam(x, y):
	import numpy
	xs = numpy.array(x)
	ys = numpy.array(y)
	return sum(abs(xs - ys))

def euclidean(x,y):
	import numpy
	from math import sqrt
	xs = numpy.array(x)
	ys = numpy.array(y)
	diff_sqrd = numpy.array((xs - ys)**2)
	return sqrt(sum(diff_sqrd) if diff_sqrd.size > 1 else diff_sqrd)

def supreme(x,y):
	import numpy
	xs = numpy.array(x)
	ys = numpy.array(y)
	return max(abs(xs - ys))