# Generates the datapoints for admUI and cvxopt in Figure 3 of the paper
from admUI import computeQUI
import cvxopt_solve
import dit
import time
import numpy as np
import scipy.io
import cvxopt.solvers
import argparse

parser = argparse.ArgumentParser(description='Test admUI vs. cvxopt.')
parser.add_argument('cvsfilename', nargs="?", default="",
                    help=("A cvsfile name.  "
                          "If omitted, results are output to stdout."))
parser.add_argument('-nsmax', nargs=1, default=10,
                    help=("The largest cardinality for S "
                          "for which to run the tests."))
parser.add_argument('-ndist', nargs=1, default=100,
                    help=("How many distributions to test "
                          "for each value of ns."))
args = parser.parse_args()
cvsfilename = args.cvsfilename
# test configuration
nsmax = args.nsmax
ndist = args.ndist

logging = cvsfilename != ''
if logging:
    print("Logging results to file '{}'".format(cvsfilename))
    cvsfile = open(cvsfilename, "w")
    cvsfile.write(
        "# ns, admUI_result, admUI_time, cvxopt_result, cvxopt_time\n")

# Silence the cvsopt solver:
cvxopt.solvers.options['show_progress'] = False


def cvxopt_solve_PDF(pdf):
    '''Call the solver from cvxopt_solve and compute UI from its output.'''
    p_xy = cvxopt_solve.marginal_xy(pdf)
    p_xz = cvxopt_solve.marginal_xz(pdf)
    p_z = dict()
    for xz, prob in p_xz.items():
        x, z = xz
        if z in p_z.keys():
            p_z[z] += prob
        else:
            p_z[z] = prob
    cvx = cvxopt_solve.Cvxopt_Solve(p_xy, p_xz)
    cvx.solve_it()
    Qs = dict()
    for xyz, i in cvx.var_idx.items():
            Qs[xyz] = cvx.solver_ret['x'][i]
    p_yz = cvxopt_solve.marginal_yz(Qs)
    I_XY_Z = 0
    for xyz, t in Qs.items():
        x, y, z = xyz
        if t > 0:
            I_XY_Z += t * cvxopt.log((t * p_z[z])/(p_xz[(x, z)]*p_yz[(y, z)]))
    return I_XY_Z / cvxopt.log(2)


# read data file
mat = scipy.io.loadmat('../data/dataPs.mat')
npy = np.array(mat['Ps'])

UIv = np.empty(shape=(ndist, nsmax))
ltimev = np.empty(shape=(ndist, nsmax))
UIcv = np.empty(shape=(ndist, nsmax))
ltimecv = np.empty(shape=(ndist, nsmax))

for ns in range(1, nsmax):
    ny = ns
    nz = ns
    print("--------------- ns= %s ---------------" % (ns + 1))
    for i in range(0, ndist):
        Pt = npy[:, i, ns]
        P = Pt[Pt != 0]
        Ps = P.reshape(nz + 1, ny + 1, ns + 1)
        d = dit.Distribution.from_ndarray(Ps)
        d.set_rv_names('SXY')
        print(i)

        # admUI
        start_time = time.time()
        Q = computeQUI(distSXY=d)  # , DEBUG=True)
        UIX = (dit.shannon.conditional_entropy(Q, 'S', 'Y')
               + dit.shannon.conditional_entropy(Q, 'X', 'Y')
               - dit.shannon.conditional_entropy(Q, 'SX', 'Y'))
        lapsed_time = time.time() - start_time
        UIv[i, ns] = UIX
        ltimev[i, ns] = lapsed_time
        if logging:
            cvsfile.write("{}, ".format(ns + 1))
            cvsfile.write("{:.15f}, {:.15f}, ".format(UIX, lapsed_time))
        else:
            print("admUI = %.15f" % UIX, "      time = %.15f" % lapsed_time)

        # cvxopt
        pdf = dict(zip(d.outcomes, d.pmf))
        start_timec = time.time()
        UIXc = cvxopt_solve_PDF(pdf)
        lapsed_timec = time.time() - start_timec
        UIcv[i, ns] = UIXc
        ltimecv[i, ns] = lapsed_timec
        if logging:
            cvsfile.write("{:.15f}, {:.15f}\n".format(UIXc, lapsed_timec))
        else:
            print("cvxUI = %.15f" % UIXc, "      time = %.15f" % lapsed_timec)
    print('')

np.set_printoptions(precision=15)
print("-------------------- admUI --------------------")
UIv = np.delete(UIv, 0, 1)
mUIv = np.mean(UIv, axis=0)
ltimev = np.delete(ltimev, 0, 1)
mltimev = np.mean(ltimev, axis=0)
print("admUI:")
print(mUIv)
print("time:")
print(mltimev)

np.set_printoptions(precision=15)
print("-------------------- cvxUI --------------------")
UIcv = np.delete(UIcv, 0, 1)
mUIcv = np.mean(UIcv, axis=0)
ltimecv = np.delete(ltimecv, 0, 1)
mltimecv = np.mean(ltimecv, axis=0)
print("cvxUI:")
print(mUIcv)
print("time:")
print(mltimecv)

cvsfile.close()
