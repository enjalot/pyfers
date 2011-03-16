import simple_pure
import pyximport; pyximport.install()
import simple_cython
#import simple_cl
#import simple_swig
import numpy

import timing

if __name__ == "__main__":
    n = 1000

    a = numpy.ones((n, 1), dtype=numpy.float32)
    b = numpy.ones((n, 1), dtype=numpy.float32)

    cpure = simple_pure.add(a, b)
    cnp = simple_pure.add_np(a, b)
    cyp = simple_cython.add1(a, b)

    print timing.timings

