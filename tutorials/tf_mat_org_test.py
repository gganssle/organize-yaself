'''
This son of a gun tests the restructuring of TF tensors for an SOM project.

bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
tf.pow(tf.sub(self._weightage_vects, tf.pack(
    [self._vect_input for i in range(m*n)])), 2), 1)),
                  0)

RESULTS in above: 2 = the exp for tf.pow, 
                  1 = sum across the RGB channels, 
                  0 = min index across the 1D (0th dimension) vector

Hey also, these are just notes, so none of this stuff runs together or anything like that. I'm just using pieces of this script in various situations.

'''

import numpy as np
import tensorflow as tf

#### 2D output / 1D input example #####################################################
'''
ns = 4
dim = 3

m = 8
n = 9

temp = np.zeros((ns,dim))
for i in range(ns):
	for j in range(dim):
		temp[i,j] = i*10 + j

weights = tf.random_normal([m*n, dim])
inp = tf.constant(temp[0], dtype=tf.float32)

bmuIdx = tf.argmin(tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(weights, tf.pack([inp for i in range(m*n)])), 2), 1)), 0)
'''
### 3D output / 1D input example #####################################################
ns = 4
dim = 3

l = 7
m = 8
n = 9

temp = np.zeros((ns,dim))
for i in range(ns):
	for j in range(dim):
		temp[i,j] = i*10 + j

#print('original = \n', temp, '\n')

weights = np.zeros((l*m*n, dim))
for i in range(l*m*n):
	for j in range(dim):
		weights[i,j] = (i - (l*m*n/2))**2 + j

#weights = tf.random_normal([l*m*n, dim])
inp = tf.constant(temp[0])

bmuIdx = tf.argmin(tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(weights, tf.pack([inp for i in range(l*m*n)])), 2), 1)), 0)

### neuron locations class ######################################################
#3D output
def neuron_locations(l, m, n):
	for k in range(l):
		for i in range(m):
			for j in range(n):
				yield np.array([k, i, j])
 
#2D output
'''
def neuron_locations(m, n):
	for i in range(m):
		for j in range(n):
			yield np.array([i, j])
'''
### slice stuff #################################################################
#3D output
locs = tf.constant(np.array(list(neuron_locations(l, m, n))))

slce = tf.pad(tf.reshape(bmuIdx, [1]), np.array([[0,1]]))

bmu_loc = tf.reshape(tf.slice(locs, slce, tf.constant(np.array([1, 3]))), [3])

#2D output 
'''
locs = tf.constant(np.array(list(neuron_locations(m, n))))

slce = tf.pad(tf.reshape(bmuIdx, [1]), np.array([[0,1]]))

bmu_loc = tf.reshape(tf.slice(locs, slce, tf.constant(np.array([1, 2]))), [2])
'''

sess = tf.Session()
print('bmu index = ', sess.run(bmuIdx))
print('slice = ', sess.run(slce))
#print(sess.run(locs))
print('bmu location = ', sess.run(bmu_loc))



