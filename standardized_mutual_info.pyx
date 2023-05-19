#cython: language_level=3
import cython
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from cpython.exc cimport PyErr_CheckSignals

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# INTTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
INTTYPE = np.int64
FLOATTYPE = np.float64

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int64_t INTTYPE_t
ctypedef np.float64_t FLOATTYPE_t

cdef extern from "math.h":
    FLOATTYPE_t log(FLOATTYPE_t x) nogil
    FLOATTYPE_t exp(FLOATTYPE_t x) nogil
    FLOATTYPE_t lgamma(FLOATTYPE_t x) nogil
    FLOATTYPE_t sqrt(FLOATTYPE_t x) nogil

cdef FLOATTYPE_t _log(FLOATTYPE_t x) nogil:
    """Compute the log of x, but return 0 if x is 0."""
    if x == 0:
        return 0
    return log(x)

@cython.cdivision(True)
cdef FLOATTYPE_t _getP(INTTYPE_t a, INTTYPE_t b,INTTYPE_t N) nogil:
    """Gets the the probability for the smallest number of successes n for Hyp(a,b,N).
    See also https://en.wikipedia.org/wiki/Hypergeometric_distribution.
    
    Args:
        a (int): Number of objects that count as success.
        b (int): Number of draws.
        N (int): Total number of objects.
    
    Returns:
        float: The probability of the smallest n for Hyp(a,b,N).
    """
    cdef INTTYPE_t nij = max(0, a + b - N)
    return exp(lgamma(a + 1) - lgamma(a - nij + 1) + lgamma(b + 1) - lgamma(b - nij + 1) + lgamma(N - b + 1) - lgamma(nij + 1) - lgamma(N - a - b + nij + 1) - lgamma(N + 1) + lgamma(N - a + 1))

@cython.cdivision(True)
cdef FLOATTYPE_t _incrP(FLOATTYPE_t p, INTTYPE_t a, INTTYPE_t b, INTTYPE_t n, INTTYPE_t N) nogil:
    """Given the probability of n successes for a Hyp(a,b,N) computes the probability of n+1 successes."""
    return p * (a - n) * (b - n) / (n + 1) / (N - a - b + n + 1)

@cython.cdivision(True)
@cython.boundscheck(False)
def standardized_mutual_info_cython(contingency, INTTYPE_t N):
    """Standardized mutual information for two clusterings under pairwise permutations.

    This code is based on the paper "Standardized Mutual Information for Clustering
    Comparisons: One Step Further in Adjustment for Chance" by S. Romano et al.
    (https://proceedings.mlr.press/v32/romano14.html).

    Args:
        contingency: Sparse contingency matrix of shape (n_classes_true, n_classes_pred).

    Returns:
        The standardized mutual information.
    """
    cdef np.ndarray a_arr = np.ravel(contingency.sum(axis=0))
    cdef np.ndarray b_arr = np.ravel(contingency.sum(axis=1))

    cdef INTTYPE_t r = a_arr.shape[0]
    cdef INTTYPE_t c = b_arr.shape[0]

    if c > r:
        a_arr, b_arr = b_arr, a_arr
        r, c = c, r

    cdef FLOATTYPE_t sum_nLogn = contingency.data.dot(
        np.where(contingency.data == 0, 0, np.log(contingency.data)))

    # Brute Force summation
    cdef INTTYPE_t[:] a = a_arr
    cdef INTTYPE_t[:] b = b_arr
    cdef np.ndarray EP_arr = np.zeros((r, c), dtype=FLOATTYPE)
    cdef FLOATTYPE_t[:,:] EP = EP_arr
    cdef INTTYPE_t i, j, nij, N_, a_, jp, b_, nijp, ip, nipjp, nipj
    cdef FLOATTYPE_t sumP, sumP_, Lpnij, p, p_, p__
    cdef FLOATTYPE_t E_sum_nLogn = 0

    for i in range(r):
        for j in range(c):
            p = _getP(a[i], b[j], N)

            for nij in range(max(0, a[i] + b[j] - N), min(a[i], b[j]) + 1):
                E_sum_nLogn = E_sum_nLogn + nij * _log(nij) * p
                sumP = 0

                N_ = N - b[j]
                a_ = a[i] - nij

                for jp in range(j + 1, c):
                    b_ = b[jp]
                    p_ = _getP(a_, b_, N_)

                    for nijp in range(max(0, a_ + b_ - N_), min(a_, b_) + 1):
                        sumP_ = 0
                        for ip in range(r):
                            if ip == i:
                                continue
                            p__ = _getP(a[ip], b[jp] - nijp, N - a[i])
                            for nipjp in range(max(0, a[ip] + b[jp] - nijp - N + a[i]), min(a[ip], b[jp]) + 1):
                                sumP_ = sumP_ + nipjp * \
                                    _log(nipjp) * p__
                                p__ = _incrP(p__, a[ip], b[jp] -
                                             nijp, nipjp, N - a[i])

                        sumP_ = sumP_ + nijp * _log(nijp)

                        sumP = sumP + sumP_ * p_
                        p_ = _incrP(p_, a_, b_, nijp, N_)

                        # Check for signals
                        PyErr_CheckSignals()

                N_ = N - a[i]
                b_ = b[j] - nij
                for ip in range(i + 1, r):
                    a_ = a[ip]
                    p_ = _getP(a_, b_, N_)
                    for nipj in range(max(0, a_ + b_ - N_), min(a_, b_) + 1):
                        sumP = sumP + nipj * _log(nipj) * p_
                        p_ = _incrP(p_, a_, b_, nipj, N_)
                    
                # Check for signals
                PyErr_CheckSignals()

                sumP = 2 * sumP + nij * _log(nij)

                Lpnij = nij * _log(nij) * p
                EP[i, j] = EP[i, j] + Lpnij * sumP
                p = _incrP(p, a[i], b[j], nij, N)

    cdef FLOATTYPE_t E_sum_nLogn_2 = EP_arr.sum()
    cdef FLOATTYPE_t variance = E_sum_nLogn_2 - E_sum_nLogn**2

    # Analytical continuation for zero variance (accounting for numerical errors)
    if variance < 1e-10:
        # If there is no variance, the expected value is the observed value
        # such that the analytical continuation is 1.0
        return 1.0

    return (sum_nLogn - E_sum_nLogn) / sqrt(variance)