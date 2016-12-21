'''
This code is adapted from https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
'''

import tensorflow as tf
import numpy as np
 
 
class SOM(object): 
    _trained = False
 
    def __init__(self, l, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        self._m = m
        self._n = n
        self._l = l
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(l, m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))
 
        self._graph = tf.Graph()
 
        with self._graph.as_default():
            self._weightage_vects = tf.Variable(tf.random_normal(
                [l*m*n, dim]))
 
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(l, m, n))))
 
            self._vect_input = tf.placeholder("float", [dim])
            self._iter_input = tf.placeholder("float")

            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.sub(self._weightage_vects, tf.pack(
                    [self._vect_input for i in range(l*m*n)])), 2), 1)),
                                  0)
 
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]))),
                                 [2])
 
            learning_rate_op = tf.sub(1.0, tf.div(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.mul(alpha, learning_rate_op)
            _sigma_op = tf.mul(sigma, learning_rate_op)


            print('\n--------------------------------\n')


            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.sub(
                self._location_vects, tf.pack(
                    [bmu_loc for i in range(l*m*n)])), 2), 1)
            print('\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n')
            neighbourhood_func = tf.exp(tf.neg(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.mul(_alpha_op, neighbourhood_func)


            print('\n++++++++++++++++++++++++++++\n')


            learning_rate_multiplier = tf.pack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                               for i in range(l*m*n)])
            weightage_delta = tf.mul(
                learning_rate_multiplier,
                tf.sub(tf.pack([self._vect_input for i in range(l*m*n)]),
                       self._weightage_vects))                                         
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)                                       
 
            self._sess = tf.Session()
 
            init_op = tf.initialize_all_variables()
            self._sess.run(init_op)
 
    def _neuron_locations(self, l, m, n):
        for k in range(l):
            for i in range(m):
                for j in range(n):
                    yield np.array([k, i, j])
 
    def train(self, input_vects):
        for iter_no in range(self._n_iterations):
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})
 
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid
 
        self._trained = True
 
    def get_centroids(self):
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid
 
    def map_vects(self, input_vects): 
        if not self._trained:
            raise ValueError("SOM not trained yet")
 
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect-
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])
 
        return to_return

###########################################################################
from matplotlib import pyplot as plt
 
colors = np.array(
     [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])
color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']
 
som = SOM(20, 30, 3, 400)
som.train(colors)
 
image_grid = som.get_centroids()
 
mapped = som.map_vects(colors)
 
plt.imshow(image_grid)
plt.title('Color SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], color_names[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show()
