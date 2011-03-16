#http://wiki.cython.org/tutorials/numpy
#http://docs.cython.org/src/tutorial/numpy.html

import numpy
cimport numpy

import timing

@timing.timings
def add1(numpy.ndarray[numpy.float32_t, ndim=1] a, 
         numpy.ndarray[numpy.float32_t, ndim=1] b):
    
    cdef numpy.ndarray[numpy.float32_t, ndim=1] c = numpy.empty_like(a)#numpy.zeros(a.shape, dtype=numpy.float32)
    for i in range(0, a.shape[0]):
        c[i] = a[i] + b[i]
    return c


