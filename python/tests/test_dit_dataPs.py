# test_dit_dataPs.py
# This file compares admUI with the algorithm implemented in dit
# on the distributions in dataPs.mat (see directory /data/).
# The results are used in Figure 3 of the paper.
#
# Distributed under the terms of the GNU General Public License v3
from admUI import computeQUI
import dit
import time
import numpy as np
import scipy.io

# read data file
mat = scipy.io.loadmat('../data/dataPs.mat')
npy = np.array(mat['Ps'])

# test configuration
# (for ns > 5, dit takes inordinately long for some distributions)
nsmax = 5
ndist = 3  # 100

UIv = np.empty(shape=(ndist, nsmax))
ltimev = np.empty(shape=(ndist, nsmax))
UIpv = np.empty(shape=(ndist, nsmax))
ltimepv = np.empty(shape=(ndist, nsmax))
dit_errorcnt = 0
UIcv = np.empty(shape=(ndist, nsmax))
ltimecv = np.empty(shape=(ndist, nsmax))

for ns in range(1, nsmax):
    ny = ns
    nz = ns
    print("--------------- ns = %s ---------------" % (ns + 1))
    for i in range(0, ndist):
        Pt = npy[:, i, ns]
        P = Pt[Pt != 0]
        Ps = P.reshape(nz + 1, ny + 1, ns + 1)
        d = dit.Distribution.from_ndarray(Ps)
        d.set_rv_names('SXY')

        # admUI
        start_time = time.time()
        Q = computeQUI(d)
        UIX = (dit.shannon.conditional_entropy(Q, 'S', 'Y')
               + dit.shannon.conditional_entropy(Q, 'X', 'Y')
               - dit.shannon.conditional_entropy(Q, 'SX', 'Y'))
        lapsed_time = time.time() - start_time
        UIv[i, ns] = UIX
        ltimev[i, ns] = lapsed_time

        # dit
        start_timep = time.time()
        try:
            dit_pid = dit.pid.PID_BROJA(d, ['X', 'Y'], 'S')
        except dit.exceptions.ditException:
            print(i, "ditException: P = ", P, ", ns=ny=nz=", ns + 1, ", i=", i)
            dit_errorcnt = dit_errorcnt + 1
            UIpv[i, ns] = 0
            ltimepv[i, ns] = 0
            continue

        lapsed_timep = time.time() - start_timep
        UIXp = dit_pid.get_partial((('X', ), ))
        UIpv[i, ns] = UIXp
        ltimepv[i, ns] = lapsed_timep
        print(str(i) + ": admUI = %.15f, ditUI = %.15f" % (UIX, UIXp)
              + ", time = %.15f/%.15f" % (lapsed_time, lapsed_timep))

np.set_printoptions(precision=15)
print("-------------------- ditUI --------------------")
UIv = np.delete(UIv, 0, 1)
mUIv = np.mean(UIv, axis=0)
ltimev = np.delete(ltimev, 0, 1)
mltimev = np.mean(ltimev, axis=0)
print("admUI: " + str(mUIv))
print("time: " + str(mltimev))
print("-------------------- ditUI --------------------")
UIpv = np.delete(UIpv, 0, 1)
mUIpv = np.mean(UIpv, axis=0)
ltimepv = np.delete(ltimepv, 0, 1)
mltimepv = np.mean(ltimepv, axis=0)
print("ditUI: " + str(mUIpv))
print("time: " + str(mltimepv))
print("ditException count: %s" % dit_errorcnt)
