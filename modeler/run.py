'''
This script runs a Rock Properties Catalog lookup, prestack modeling,
and stacking operation.
'''

import mod_func
import rpc as rock

rpc = rock.RPC()
mod = mod_func.modeler()

# load rock properties
filters = ["[[lithology::Shale||Sandstone||Limestone]][[Delta::%2B]]"]
properties = ['Citation', 'Description', 'Lithology', 'Vp', 'Vs', 'Rho', 'Delta', 'Epsilon']
options = ["limit=100"]

df = rpc.query(filters, properties, options)

#df.head()
#print(df.Vs[2])

# model seismic from roscks
nang = 30	# # of offset angles
dang = 1	# offset angle increment
pf = 10		# peak freq (Hz)
dt = 0.004	# sample rate (s)
vp1 = df.Vp[2]	# top material Vp
vs1 = df.Vs[2]	# top material Vs
ro1 = df.Rho[2]	# top material dens
vp2 = df.Vp[4]	# bot material Vp
vs2 = df.Vs[4]	# bot material Vs
ro2 = df.Rho[4]	# bot material dens

print('\n', nang, dang, pf, dt, vp1, vs1, ro1, vp2, vs2, ro2, '\n')
if ro1 == None:
	ro1 = 2.2
if ro2 == None:
	ro2 = 2.3
print('\n', nang, dang, pf, dt, vp1, vs1, ro1, vp2, vs2, ro2, '\n')

trace = mod.doit(nang, dang, pf, dt, vp1, vs1, ro1, vp2, vs2, ro2)

import matplotlib.pyplot as plt
plt.plot(trace)
plt.show()


