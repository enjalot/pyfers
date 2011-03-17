import numpy
import timing

@timing.timings("Add: Numpy")
def add_np(a, b):
    c = a + b
    return c


@timing.timings("Add: Python")
def add(a, b):
    c = numpy.empty_like(a)
    for i in xrange(0, a.shape[0]):
        c[i] = a[i] + b[i]
    return c


