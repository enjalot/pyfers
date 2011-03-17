import simple_pure
import pyximport; pyximport.install()
import simple_cython
import simple_cl
#import simple_swig
import numpy

import timing

if __name__ == "__main__":
    n = 10000000

    a = numpy.ones((n, ), dtype=numpy.float32)
    b = numpy.ones((n, ), dtype=numpy.float32)

    cpure = simple_pure.add(a, b)
    for i in range(10):
        cnp = simple_pure.add_np(a, b)
        cyp = simple_cython.add(a, b)
        clp = simple_cl.add(a, b)

        ccy = simple_cython.complex(a, b)
        ccl = simple_cl.complex(a, b)
    #cyp = simple_cython.add2(a, b)
    print numpy.linalg.norm(cnp - clp)
    print numpy.linalg.norm(ccy - ccl)

    #cyc = simple_cython.gradient

    print "n = %d" % n
    print timing.timings

