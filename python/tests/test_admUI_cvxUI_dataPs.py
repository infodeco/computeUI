# Generates the datapoints for admUI and cvxopt in Figure 3 of the paper
import sys
sys.path.insert(0, '../')
from admUI import computeQUI
from cvxopt_solve import *
from dit import *
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

# mirror sys.stdout to a log file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logPs_admUI_cvxopt.dat", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

sys.stdout = Logger()

# read data file
mat = scipy.io.loadmat('../../data/dataPs.mat')
npy = np.array(mat['Ps'])

# test configuration 
nsmax = 10
ndist = 100

UIv = np.empty(shape=(ndist,nsmax))
ltimev = np.empty(shape=(ndist,nsmax))
UIcv = np.empty(shape=(ndist,nsmax))
ltimecv = np.empty(shape=(ndist,nsmax))

for ns in xrange(1,nsmax):
    ny=ns; nz=ns;
    print("--------------- ns= %s ---------------" %(ns+1))
    for i in xrange(0,ndist):
        Pt = npy[:,i,ns]
	P  = Pt[Pt!=0]
	Ps = P.reshape(nz+1,ny+1,ns+1)
	d = Distribution.from_ndarray(Ps)
        d.set_rv_names('SXY')
	print(i)
	
	# admUI
	start_time = time.time()
	Q = computeQUI(distSXY = d, DEBUG = True)
        UIX = dit.shannon.conditional_entropy(Q, 'S', 'Y') + dit.shannon.conditional_entropy(Q, 'X', 'Y') - dit.shannon.conditional_entropy(Q, 'SX', 'Y')
	lapsed_time = time.time() - start_time
	UIv[i,ns] = UIX
	ltimev[i,ns] = lapsed_time 
	print("admUI = %.15f" %UIX, "      time = %.15f" %lapsed_time)

	# cvxopt
	pdf=dict(zip(d.outcomes, d.pmf))
	start_timec = time.time()
	UIXc=solve_PDF(pdf)
	lapsed_timec = time.time() - start_timec
	UIcv[i,ns] = UIXc
	ltimecv[i,ns] = lapsed_timec 
	print("cvxUI = %.15f" %UIXc, "      time = %.15f" %lapsed_timec)

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

np.set_printoptions(precision=15)
print("-------------------- cvxUI --------------------")
UIcv = np.delete(UIcv,0,1)
mUIcv = np.mean(UIcv,axis=0)
ltimecv = np.delete(ltimecv,0,1)
mltimecv = np.mean(ltimecv,axis=0)
print("cvxUI:")
print(mUIcv)
print("time:")
print(mltimecv)








