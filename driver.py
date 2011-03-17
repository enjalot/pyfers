import simple_pure
import pyximport; pyximport.install()
import simple_cython
import simple_cl
#import simple_swig
import numpy

import timing

if __name__ == "__main__":
    n = 100000

    a = numpy.ones((n, ), dtype=numpy.float32)
    b = numpy.ones((n, ), dtype=numpy.float32)

    cpure = simple_pure.add(a, b)
    for i in range(10):
        cnp = simple_pure.add_np(a, b)
        cyp1 = simple_cython.add_cy1(a, b)
        cyp2 = simple_cython.add_cy2(a, b)
        clp = simple_cl.add_cl(a, b)
    #cyp = simple_cython.add2(a, b)

    print timing.timings

