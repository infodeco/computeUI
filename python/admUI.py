import dit
import numpy
from admUI_numpy import computeQUI_numpy 


def computeQUI(distSXY, eps = 1e-7, DEBUG = False, IPmethod = "IS"):
    '''
    Compute an optimizer Q

    distSXY : A joint distribution of three variables (as a dit.Distribution).
    eps     : The precision of the outer loop.  The precision of the inner loop will be eps / (20 |S|).
    DEBUG   : Print output for debugging.

    The computation is carried out using computeQUI_numpy
    '''
    # Prepare distributions
    QSXYd = distSXY.copy()    # make a copy, to not overwrite the argument
    QSXYd.set_rv_names('SXY')
    QSXYd.make_dense()        # the set of outcomes should be Cartesian

    # collect state spaces / cardinalities
    suppS = QSXYd.alphabet[0]
    nS = len(suppS)
    suppX = QSXYd.alphabet[1]
    nX = len(suppX)
    suppY = QSXYd.alphabet[2]
    nY = len(suppY)
    # to do: take relevant subset of suppS

    samplespace = QSXYd.outcomes
    QSXYa = QSXYd.pmf.reshape(nS, nX, nY)

    PS = QSXYd.marginal('S').pmf.reshape(nS, 1) # PS is a column vector, for convenience

    PXgSa = numpy.array(list(map(lambda x: x.pmf, QSXYd.condition_on('S', rvs = 'X')[1]))).transpose()
    PYgSa = numpy.array(list(map(lambda x: x.pmf, QSXYd.condition_on('S', rvs = 'Y')[1]))).transpose()

    # print(1e-6 * numpy.ones((nX, nY)) / (nX * nY) + (1 - 1e-6) * QSXYd.marginal('X').pmf.reshape(nX, 1) * QSXYd.marginal('Y').pmf.reshape(1, nY))
    QSXYa = computeQUI_numpy(PXgSa, PYgSa, PS, eps = eps, DEBUG = DEBUG, IPmethod = IPmethod).reshape(-1)

    QSXYd = dit.Distribution(samplespace, QSXYa)
    QSXYd.set_rv_names('SXY')
    return QSXYd
