'''
This script runs a Rock Properties Catalog lookup, prestack modeling,
and stacking operation for a set of prototype vecors and saves them in ../pvs
'''

import mod_func
import rpc as rock

rpc = rock.RPC()
mod = mod_func.modeler()

# init
rknum = 10	# number of rocks per lithology
rkprop = []	# list 4 rock properties
nang = 30	# # of offset angles
dang = 1	# offset angle increment
pf = 10		# peak freq (Hz)
dt = 0.004	# sample rate (s)

# load rock properties
	# sands
filters = ["[[lithology::sandstone]][[vp::>3000]][[vp::<5000]]"]
properties = ['Vp', 'Vs', 'Rho']
options = [''.join(("limit=",str(rknum)))]

df = rpc.query(filters, properties, options)

for i in range(rknum):
	if df.Rho[i] == None:
		df.Rho = 2.2
	rkprop.append([df.Vp[i], df.Vs[i], df.Rho[i]])

	# shales
filters = ["[[lithology::shale]][[vp::>4000]][[vp::<6000]]"]
properties = ['Vp', 'Vs', 'Rho']
options = ["limit=10"]

df = rpc.query(filters, properties, options)

for i in range(rknum):
	if df.Rho[i] == None:
		df.Rho = 2.4
	rkprop.append([df.Vp[i], df.Vs[i], df.Rho[i]])

	# limes
filters = ["[[lithology::limestone]][[vp::>4000]][[vp::<6000]]"]
properties = ['Vp', 'Vs', 'Rho']
options = ["limit=10"]

df = rpc.query(filters, properties, options)

for i in range(rknum):
	if df.Rho[i] == None:
		df.Rho = 2.4
	rkprop.append([df.Vp[i], df.Vs[i], df.Rho[i]])

# model all interface possibilities
f = open('../pvs/20170103.wavelets', 'w')
for i in range(rknum*3):
	for j in range(rknum*3):
		if i != j: # there is no interface if rocks are same
			vp1 = rkprop[i][0]	# top material Vp
			vs1 = rkprop[i][1]	# top material Vs
			ro1 = rkprop[i][2]	# top material dens
			vp2 = rkprop[j][0]	# bot material Vp
			vs2 = rkprop[j][1]	# bot material Vs
			ro2 = rkprop[j][2]	# bot material dens

			trace = mod.doit(nang, dang, pf, dt, vp1, vs1, ro1, vp2, vs2, ro2)

			for k in range(len(trace)):
				f.write(str(trace[k]))
				f.write(' ')
			f.write('\n')

f.close()




