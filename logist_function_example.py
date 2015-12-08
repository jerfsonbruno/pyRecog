import matplotlib.pyplot as plt
import numpy as np
from math import exp
def f(x):
	return 1.0/(1.0+exp(-4*(x-0)))

X = np.linspace(-1,1,20)
T = np.array(list(map(f, X)))
plt.plot(X,T,'b')
#plt.show()
X = np.asmatrix(X)
T = np.asmatrix(T)
t_set = np.concatenate((X,T), axis=0).T
train = []
for i in range(len(t_set)):
	train += [np.array(t_set)[i].tolist()]

import pyRecog
r = pyRecog.RBF(3,1)
r.train(train, 0.0, 100000, 0.1, 0.5)

X = np.linspace(-1,1,20)
Y = []
for i in X:
	Y += [r([i])[0]]

plt.plot(X,Y,'r')