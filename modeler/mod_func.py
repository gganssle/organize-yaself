'''
This script builds several wavelets using the Aki-Richards
approximation to the Zoeppritz equations. The modeled
wavelets will be used as prototype vectors as training
inputs to the post-stack seismic SOM.
'''

import numpy as np
import math

class modeler(object):

	def __init__(self):

		pass

	# Aki-Richards equation
	def aki(self, thta, vp1, vs1, ro1, vp2, vs2, ro2):
		rad = thta * math.pi / 180

		dvp = vp2 - vp1
		dvs = vs2 - vs1
		dro = ro2 - ro1

		k = (vs1 / vp1)**2

		r = (1/2)*((dvp/vp1)+(dro/ro1)) + \
			((1/2)*(dvp/vp1) - 2 * k * (2*(dvs/vs1)+(dro/ro1))) * math.sin(rad)**2 + \
			(1/2) * (dvp/vp1) * (math.tan(rad)**2 - math.sin(rad)**2)
	
		return r

	# build a ricker wavelet
	def ricker(self, pf, dt):
		nw = 2 * int(math.floor(2.2 / (pf *dt) / 2)) + 1
		nc = int(nw / 2)

		w = np.zeros(nw)
		k = np.zeros(nw)

		for i in range(nw):
			k[i] = i

		a = (nc - k + 1) * pf * dt * math.pi
		b = a**2
		w = (1 - (b*2)) * np.exp(-b)

		return w
	
	def doit(self, nang, dang, pf, dt, vp1, vs1, ro1, vp2, vs2, ro2):
		# allocate mem
		wavefield = np.zeros((2 * int(math.floor(2.2 / (pf *dt) / 2)) + 1, math.floor(nang/dang)))

		# build the offset response
		for idx1 in range(wavefield.shape[1]):
			ang = idx1 * dang
	
			rc = self.aki(ang, vp1, vs1, ro1, vp2, vs2, ro2)

			wavefield[:,idx1] = np.convolve(self.ricker(pf,dt), rc)

		# stack out a single trace
		tr = np.sum(wavefield, axis=1)

		return tr




