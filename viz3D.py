
import numpy as np
import matplotlib.pyplot as plt

with open('dat/map.ascii', 'r') as f:
    raw = f.read().split()

print(len(raw))

loaded = np.zeros((50,50,50,55))

for i in range(50):
    for j in range(50):
        for k in range(50):
            for l in range(55):
                loaded[i,j,k,l] = float(raw[l + k*55 + j*50*55 + i*50*50*55])

#plt.imshow(loaded[0,:,:,:])
#plt.show()

print(loaded[0,0,34,:])
