#http://wiki.cython.org/tutorials/numpy
#http://docs.cython.org/src/tutorial/numpy.html

import numpy
cimport numpy

import timing

@timing.timings("Add: Cython")
def add(numpy.ndarray[numpy.float32_t, ndim=1] a, 
         numpy.ndarray[numpy.float32_t, ndim=1] b):
    
    cdef numpy.ndarray[numpy.float32_t, ndim=1] c = numpy.empty_like(a)
    cdef Py_ssize_t i
    for i in range(0, a.shape[0]):
        c[i] = a[i] + b[i]
    return c

@timing.timings("Complex: Cython")
def complex(numpy.ndarray[numpy.float32_t, ndim=1] a, 
         numpy.ndarray[numpy.float32_t, ndim=1] b):
    
    cdef numpy.ndarray[numpy.float32_t, ndim=1] c = numpy.empty_like(a)
    cdef Py_ssize_t i
    cdef float ai
    cdef float bi
    for i in range(0, a.shape[0]):
        ai = a[i]
        bi = b[i]
        c[i] = ai * 3. + 2. * bi - ai*bi + 35. / bi - 25. / ai - ai*ai*ai*ai / bi / bi + 2./(bi*bi*bi);
    return c


