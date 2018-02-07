import numpy
# from numpy import *

maxiter = 1000
maxiter2 = maxiter

def computeQUI_numpy(PXgSa, PYgSa, PS, eps = 1e-7, DEBUG = False, IPmethod = "GIS",
                         maxiter = 1000, maxiter2 = 1000):
    # print(PXgSa)
    # print(PYgSa)
    nX = PXgSa.shape[0]
    nY = PYgSa.shape[0]
    nS = PXgSa.shape[1]
    nXY = nX * nY
#    nSXY = nS * nX * nY
    rangeS = range(nS)

    #############  Start the loop
    eps2 = eps / (20 * nS)

    # start with a full support
    RXYa = 1e-6 * numpy.ones((nX, nY)) / nXY + (1 - 1e-6) * numpy.outer(numpy.dot(PXgSa, PS), numpy.dot(PYgSa, PS))
    QXYgSa = numpy.zeros((nX, nY, nS))
    for it in range(maxiter):
        diff = 1.
        ##### Step 1
        for s in rangeS:
            # print(s, ":")
            # b is zero if PXgSa[x, s] == 0 or PXgSa[y, s] == 0
            if (IPmethod == "IS"):
                Ip, xindices, yindices = Iproj_tech_IS(PXgSa[:, s], PYgSa[:, s], RXYa, eps = eps2, DEBUG = DEBUG,
                                                           maxiter2 = maxiter2)
            else:
                Ip, xindices, yindices = Iproj_tech_GIS(PXgSa[:, s], PYgSa[:, s], RXYa, eps = eps2, DEBUG = DEBUG,
                                                           maxiter2 = maxiter2)
            Ip = Ip.reshape(-1)
            if (numpy.amin(QXYgSa[numpy.outer(xindices, yindices), s]) <= 0.):
                diffs = 2.
            else:
                diffs = numpy.amax(Ip / QXYgSa[numpy.outer(xindices, yindices), s])
            if (diffs > diff):
                diff = diffs

            QXYgSa[numpy.outer(xindices, yindices), s] = Ip

        ##### Step 2
        RXYa = numpy.dot(QXYgSa.reshape(nXY, nS), PS).reshape(nX, nY)
        if (diff - 1. < eps):
            break
        # else:
        #     print(diff)

    # if DEBUG:
    #     print("it: ", it)
    if (it + 1 == maxiter):
        print("Warning: Maximum number of iterations reached in outer loop.")

    QSXYa = (QXYgSa * PS[:, 0]).transpose((2, 0, 1))
    # QSXYa = QSXYa.reshape(-1)
    return QSXYa
    

def Iproj_tech_GIS(PXgsa, PYgsa, RXYa, eps = 1e-9, DEBUG = False, maxiter2 = 1000):
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
    # b = RXYa.copy()
    b = RXYa[numpy.outer(xindices, yindices)].reshape(nXi, nYi)
    rangeXb = range(nXi)
    rangeYb = range(nYi)

    factorD = numpy.sqrt(PXgsa[xindices, numpy.newaxis] * PYgsa[numpy.newaxis, yindices])  # denominator of iteration factor
    for it2 in range(maxiter2):
        # print("b:")
        # print(b)
        oosbx = numpy.sqrt(1. / numpy.sum(b, 1))
        oosby = numpy.sqrt(1. / numpy.sum(b, 0))
#                print("b:", b.shape, b)
        # bx = numpy.array(numpy.sum(b, 1)).reshape(-1)
        # by = numpy.array(numpy.sum(b, 0)).reshape(-1)

        factor = factorD * oosbx[:, numpy.newaxis] * oosby[numpy.newaxis, :]

                # for x in rangeXb:
                #     for y in rangeYb:
                #         # compute and apply the update factor
                #         if (b[x, y] != 0):
                #             facxy = PXgSa[x, s] * PYgSa[y, s]
                #             print(x, y)
                #             facxy = sqrt(facxy / (bx[x] * by[y]))
                #             print(factor[x, y])
                #             print(facxy)
#                            b[x, y] = b[x, y] * facxy
                #             # check if the update factor was larger than the previous maximum
                #             if (facxy > diff2):
                #                 diff2 = facxy

        b *= factor
        diff2 = numpy.amax(factor)
        if (diff2 < 1. + eps):
            break
        if (it2 + 1 == maxiter2):
            print("Warning: Maximum number of iterations reached in inner loop.")

#            print(b.reshape(-1) / QXYgSa[numpy.outer(xindices, yindices), s])

    # if DEBUG:
    #     print("it2: ", it2)
    return b, xindices, yindices

def Iproj_tech_IS(PXgsa, PYgsa, RXYa, eps = 1e-9, DEBUG = False, maxiter2 = 1000):
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
    rangeXb = range(nXi)
    rangeYb = range(nYi)

#    factorD = sqrt(PXgsa[xindices, newaxis] * PYgsa[newaxis, yindices])  # denominator of iteration factor
    for it2 in range(maxiter2):
        factorx = PXgsa[xindices] / numpy.sum(b, 1)
        b *= factorx[:, numpy.newaxis]
        factory = PYgsa[yindices] / numpy.sum(b, 0)
        b *= factory[numpy.newaxis, :]

        diff2 = numpy.amax(factorx) * numpy.amax(factory)
        if (diff2 < 1. + eps):
            break
        if (it2 + 1 == maxiter2):
            print("Warning: Maximum number of iterations reached in inner loop.")

#            print(b.reshape(-1) / QXYgSa[numpy.outer(xindices, yindices), s])

    # if DEBUG:
    #     print("it2: ", it2)
    return b, xindices, yindices
