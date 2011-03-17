#http://wiki.cython.org/tutorials/numpy
#http://docs.cython.org/src/tutorial/numpy.html

import numpy
cimport numpy

import timing

@timing.timings
def add_cy1(numpy.ndarray[numpy.float32_t, ndim=1] a, 
         numpy.ndarray[numpy.float32_t, ndim=1] b):
    
    cdef numpy.ndarray[numpy.float32_t, ndim=1] c = numpy.empty_like(a)
    for i in range(0, a.shape[0]):
        c[i] = a[i] + b[i]
    return c

@timing.timings
def add_cy2(numpy.ndarray[numpy.float32_t, ndim=1] a, 
         numpy.ndarray[numpy.float32_t, ndim=1] b):
    
    cdef numpy.ndarray[numpy.float32_t, ndim=1] c = numpy.empty_like(a)
    cdef Py_ssize_t i
    for i in range(0, a.shape[0]):
        c[i] = a[i] + b[i]
    return c


