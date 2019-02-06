import numpy

maxiter = 1000
maxiter2 = maxiter


def computeQUI_numpy(PXgSa, PYgSa, PS, eps=1e-7, IPmethod="GIS",
                     maxiter=1000, maxiter2=1000, DEBUG=False):
    nX = PXgSa.shape[0]
    nY = PYgSa.shape[0]
    nS = PXgSa.shape[1]
    nXY = nX * nY
    rangeS = range(nS)

    PS.reshape(-1)  # make sure that PS is a vector

    eps2 = eps / (20 * nS)

    # Start with a full support
    RXYa = (1e-6 * numpy.ones((nX, nY)) / nXY + (1 - 1e-6)
            * numpy.outer(numpy.dot(PXgSa, PS), numpy.dot(PYgSa, PS)))
    QXYgSa = numpy.zeros((nX, nY, nS))

    # -------- Start the loop
    for it in range(maxiter):
        diff = 1.
        # -------- Step 1
        for s in rangeS:
            # b is zero if PXgSa[x, s] == 0 or PXgSa[y, s] == 0
            if (IPmethod == "IS"):
                Ip, xindices, yindices = Iproj_tech_IS(
                    PXgSa[:, s], PYgSa[:, s], RXYa,
                    eps=eps2, maxiter2=maxiter2, DEBUG=DEBUG)
            else:
                Ip, xindices, yindices = Iproj_tech_GIS(
                    PXgSa[:, s], PYgSa[:, s], RXYa,
                    eps=eps2, maxiter2=maxiter2, DEBUG=DEBUG)
            Ip = Ip.reshape(-1)
            if (numpy.amin(QXYgSa[numpy.outer(xindices, yindices), s]) <= 0.):
                diffs = 2.
            else:
                diffs = numpy.amax(Ip /
                                   QXYgSa[numpy.outer(xindices, yindices), s])
            if (diffs > diff):
                diff = diffs

            QXYgSa[numpy.outer(xindices, yindices), s] = Ip

        # -------- Step 2
        RXYa = numpy.dot(QXYgSa.reshape(nXY, nS), PS).reshape(nX, nY)

        # -------- Stopping criterion
        if (diff - 1. < eps):
            break
        if DEBUG:
            print("it: ", it)
            print((QXYgSa * PS).transpose((2, 0, 1)))

    if (it + 1 == maxiter):
        print("Warning: Maximum number of iterations reached in outer loop.")

    QSXYa = (QXYgSa * PS).transpose((2, 0, 1))
    # QSXYa = QSXYa.reshape(-1)
    return QSXYa


def Iproj_tech_GIS(PXgsa, PYgsa, RXYa, eps=1e-9, maxiter2=1000, DEBUG=False):
    '''
    Generalized iterative scaling.
    '''
    nX = PXgsa.shape[0]
    nY = PYgsa.shape[0]
    rangeX = range(nX)
    rangeY = range(nY)

    xindices = [PXgsa[x] > 0 for x in rangeX]
    yindices = [PYgsa[y] > 0 for y in rangeY]
    nXi = sum(xindices)
    nYi = sum(yindices)
    b = RXYa[numpy.outer(xindices, yindices)].reshape(nXi, nYi)

    # denominator of iteration factor
    factorD = numpy.sqrt(PXgsa[xindices, numpy.newaxis]
                         * PYgsa[numpy.newaxis, yindices])
    for it2 in range(maxiter2):
        oosbx = numpy.sqrt(1. / numpy.sum(b, 1))
        oosby = numpy.sqrt(1. / numpy.sum(b, 0))

        factor = factorD * oosbx[:, numpy.newaxis] * oosby[numpy.newaxis, :]

        b *= factor
        diff2 = numpy.amax(factor)
        if (diff2 < 1. + eps):
            break
        if (it2 + 1 == maxiter2):
            print("Warning: Maximum number of iterations reached in "
                  "inner loop.")

#            print(b.reshape(-1) / QXYgSa[numpy.outer(xindices, yindices), s])
    return b, xindices, yindices


def Iproj_tech_IS(PXgsa, PYgsa, RXYa, eps=1e-9, maxiter2=1000, DEBUG=False):
    '''
    Iterative scaling.
    '''
    nX = PXgsa.shape[0]
    nY = PYgsa.shape[0]
    rangeX = range(nX)
    rangeY = range(nY)

    xindices = [PXgsa[x] > 0 for x in rangeX]
    yindices = [PYgsa[y] > 0 for y in rangeY]
    nXi = sum(xindices)
    nYi = sum(yindices)
    # b = RXYa.copy()
    b = RXYa[numpy.outer(xindices, yindices)].reshape(nXi, nYi)

# # denominator of iteration factor
#    factorD = sqrt(PXgsa[xindices, newaxis] * PYgsa[newaxis, yindices])
    for it2 in range(maxiter2):
        factorx = PXgsa[xindices] / numpy.sum(b, 1)
        b *= factorx[:, numpy.newaxis]
        factory = PYgsa[yindices] / numpy.sum(b, 0)
        b *= factory[numpy.newaxis, :]

        diff2 = numpy.amax(factorx) * numpy.amax(factory)
        if (diff2 < 1. + eps):
            break
        if (it2 + 1 == maxiter2):
            print("Warning: Maximum number of iterations reached in "
                  "inner loop.")

#            print(b.reshape(-1) / QXYgSa[numpy.outer(xindices, yindices), s])
    return b, xindices, yindices
