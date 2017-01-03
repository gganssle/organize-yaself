import som3D
import numpy as np
import matplotlib.pyplot as plt

# load prototype vectors
with open('pvs/20170103.wavelets', 'r') as f:
	raw = f.readlines()

pvs = np.zeros((len(raw), len(raw[0].split())))

for i in range(pvs.shape[0]):
	for j in range(pvs.shape[1]):
		pvs[i,j] = float(raw[i].split()[j])

# zip 'em through the SOM for training
som = som3D.SOM(50, 50, 50, pvs.shape[1], 20)

#som = sommity(50, 50, 50, pvs.shape[1], 20)

som.train(pvs)

image_grid = som.get_centroids()

print(image_grid.shape)
