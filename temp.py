 
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

            self._sess = tf.Session()
 
            init_op = tf.initialize_all_variables()
            self._sess.run(init_op)
 
    def _neuron_locations(self, l, m, n):
        for k in range(l):
            for i in range(m):
                for j in range(n):
                    yield np.array([k, i, j])



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
 
som = SOM(10, 20, 30, 3, 400)

#print(tf.Session().run(bmu_loc))
