'''
This son of a gun tests the restructuring of TF tensors for an SOM project.

bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
tf.pow(tf.sub(self._weightage_vects, tf.pack(
    [self._vect_input for i in range(m*n)])), 2), 1)),
                  0)

RESULTS in above: 2 = the exp for tf.pow, 
                  1 = sum across the RGB channels, 
                  0 = min index across the 1D (0th dimension) vector
'''

import numpy as np
import tensorflow as tf

#### 2D output / 1D input example #####################################################

ns = 4
dim = 3

m = 8
n = 9

temp = np.zeros((ns,dim))
for i in range(ns):
	for j in range(dim):
		temp[i,j] = i*10 + j

#print('original = \n', temp, '\n')

weights = tf.random_normal([m*n, dim])
inp = tf.constant(temp[0], dtype=tf.float32)

bmuIdx = tf.argmin(tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(weights, tf.pack([inp for i in range(m*n)])), 2), 1)), 0)

#sess = tf.Session()
#output = sess.run(bmuIdx)
#print(output)

### 3D output / 3D input example #####################################################
'''
ns = 4
nx = 5
ny = 5
dim = 3

l = 7
m = 8
n = 9

temp = np.zeros((dim, nx, ns, ny))
for k in range(dim):
	for q in range(nx): 
		for i in range(ns):
			for j in range(ny):
				temp[k, q, i,j] = k*1000 + q*100 + i*10 + j

#print('original = \n', temp, '\n')

weights = tf.random_normal([l*m*n, dim])
inp = tf.constant(temp[:,0,0,0], dtype=tf.float32)

bmuIdx = tf.argmin(tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(weights, tf.pack([inp for i in range(l*m*n)])), 2), 1)), 0)

sess = tf.Session()
output = sess.run(bmuIdx)
print(output)
'''
### slice stuff #################################################################
slce = tf.pad(tf.reshape(bmuIdx, [1]), np.array([[0,1]]))

bmu_loc = tf.reshape(tf.slice(self._location_vects, slce, tf.constant(np.array([1, 2]))), [2])

sess = tf.Session()
output = sess.run(bmuIdx)
print(output)
