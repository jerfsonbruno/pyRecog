# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from math import pi, exp
import numpy as np
from textwrap import wrap
import matplotlib

def f(x):
	return 1.0/(1.0+exp(-4*(x-0)))

fig = plt.figure()
ax = fig.add_subplot(111)

X2 = np.linspace(0,2*pi,400)
T2 = np.sin(X2)
X3 = np.linspace(0,2*pi,50)
noise = np.random.normal(0.0,0.1,len(X3))
T3 = np.sin(X3) + noise

X4 = np.asmatrix(X3)
T4 = np.asmatrix(T3)
t_set = np.concatenate((X4,T4), axis=0).T
train = []
for i in range(len(t_set)):
	train += [np.array(t_set)[i].tolist()]

import pyRecog
r = pyRecog.RBF(10,1)
r.train(train, 0.0, 10000, 0.1,1.0)

Y = []
sum1 = 0
for i in range(len(X2)):
	y = r([X2[i]])[0]
	Y += [y]
	sum1 += 0.5 * ( (T2[i] - y)**2 )
errm = sum1/float(len(X2))

print("\n", errm)
