import numpy
import timing

@timing.timings
def add_np(a, b):
    c = a + b
    return c


@timing.timings
def add(a, b):
    c = numpy.empty_like(a)
    for i in xrange(0, a.shape[0]):
        c[i] = a[i] + b[i]
    return c


