from computeUI import computeQUI
from dit import *
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

# mirror sys.stdout to a log file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logPy.dat", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

sys.stdout = Logger()

# read data file
mat = scipy.io.loadmat('dataPy.mat')
npy = np.array(mat['Py'])

# test configuration 
ns = 2
nz = 2 
nymax = 10
ndist = 250

UIv = np.empty(shape=(ndist,nymax))
ltimev = np.empty(shape=(ndist,nymax))
UIpv = np.empty(shape=(ndist,nymax))
ltimepv = np.empty(shape=(ndist,nymax))
dit_errorcnt = 0

for ny in xrange(1,nymax):
    print("--------------- ny= %s ---------------" %(ny+1))
    for i in xrange(0,ndist):
        Pt = npy[:,i,ny]
	P  = Pt[Pt!=0]
	Ps = P.reshape(nz,ny+1,ns)
	d = Distribution.from_ndarray(Ps)
        d.set_rv_names('SXY')
	
        try:
            pid = algorithms.pid_broja(d, ['X', 'Y'], 'S')
        except dit.exceptions.ditException:
	    print("ditException: P = ", P, ", ny=", ny+1, ", i=", i)
            dit_errorcnt = dit_errorcnt+1
	    # dummy assignments
            UIv[i,ny] = 0
	    ltimev[i,ny] = 0  
            UIpv[i,ny] = 0
	    ltimepv[i,ny] = 0  
	    continue
        else:
	    # admUI
	    start_time = time.time()
	    Q = computeQUI(distSXY = d, DEBUG = True)
            UIX = dit.shannon.conditional_entropy(Q, 'S', 'Y') + dit.shannon.conditional_entropy(Q, 'X', 'Y') - dit.shannon.conditional_entropy(Q, 'SX', 'Y')
	    lapsed_time = time.time() - start_time
	    UIv[i,ny] = UIX
	    ltimev[i,ny] = lapsed_time 
	    print("admUI = %.15f" %UIX, "      time = %.15f" %lapsed_time)

            # ditUI
	    start_timep = time.time()
            pid = algorithms.pid_broja(d, ['X', 'Y'], 'S')
	    UIXp = pid.U0
	    lapsed_timep = time.time() - start_timep
	    UIpv[i,ny] = UIXp
	    ltimepv[i,ny] = lapsed_timep 
	    print("ditUI = %.15f" %UIXp, "      time = %.15f" %lapsed_timep)

	   
np.set_printoptions(precision=15)
print("-------------------- admUI --------------------")
UIv = np.delete(UIv,0,1)
mUIv = np.mean(UIv,axis=0)
ltimev = np.delete(ltimev,0,1)
mltimev = np.mean(ltimev,axis=0)
print("admUI:")
print(mUIv)
print("time:")
print(mltimev)

print("-------------------- ditUI --------------------")
UIpv = np.delete(UIpv,0,1)
mUIpv = np.mean(UIpv,axis=0)
ltimepv = np.delete(ltimepv,0,1)
mltimepv = np.mean(ltimepv,axis=0)
print("ditUI:")
print(mUIpv)
print("time:")
print(mltimepv)
print("ditException count: %s" %dit_errorcnt)







