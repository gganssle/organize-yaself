# SOM on seismic attributes

import numpy as np
import tensorflow as tf

# initialize vars
nx = 55
ny = 49
ns = 501
dim = 3
alpha = 0.3
sigma = max(nx, ny, ns) / 2.0
n_iterations = 100

ftrs = np.zeros((dim, ns, nx, ny))

# load seismic attributes
with open('dat/amp.rsf@', 'rb') as f:
	amp = f.read(nx*ny*ns)
with open('dat/freq.rsf@', 'rb') as f:
	freq = f.read(nx*ny*ns)
with open('dat/phase.rsf@', 'rb') as f:
	phase = f.read(nx*ny*ns)

for k in range(ny):
	for j in range(nx):
		for i in range(ns):
			ftrs[0, i, j, k] = amp[i + (j * ns) + (k * ns)]
			ftrs[1, i, j, k] = freq[i + (j * ns) + (k * ns)]
			ftrs[2, i, j, k] = phase[i + (j * ns) + (k * ns)]

# build the graph
	# randomize initial weights
weights = tf.Variable(tf.random_normal([nx*ny*ns, dim])

	# SOM grid locations
locs = tf.constant(np.array(list(n_locs(nx, ny, ns))))

	# initialize training input vectors
fvec = tf.placeholder("float", [dim]) ##################################wrong########

# BMU index
bmuIdx = tf.argmin(tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(weights, tf.pack([fvec for i in range(l*m*n)])), 2), 1)), 0) ###########dimensionality wrong here#########

print(bmuIdx.shape)


n_locs(l, m, n):
	for i in range(l):
		for j in range(m):
			for k in range(n):
				yield np.array([l,m,n])
