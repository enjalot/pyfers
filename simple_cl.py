#http://documen.tician.de/pyopencl/

import pyopencl as cl
import numpy
import timing

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

prg = cl.Program(ctx, """
    __kernel void sum(__global const float *a,
                      __global const float *b, 
                      __global float *c
                      )
    {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
    """).build()

@timing.timings("Add: PyOpenCL")
def add(a, b):
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)
    
    global_size = a.shape
    local_size = None
    prg.sum(queue, global_size, local_size, a_buf, b_buf, dest_buf)
    c = numpy.empty_like(a)
    cl.enqueue_read_buffer(queue, dest_buf, c).wait()
    return c



#this kernel matches our more complex cython example
prg2 = cl.Program(ctx, """
    __kernel void complex(__global const float *a,
                      __global const float *b, 
                      __global float *c
                      )
    {
        int i = get_global_id(0);
        int ai = a[i];
        int bi = b[i];
        c[i] = ai * 3. + 2. * bi - ai*bi + 35. / bi - 25. / ai - ai*ai*ai*ai / bi / bi + 2./(bi*bi*bi);
    }
    """).build()

@timing.timings("Complex: PyOpenCL")
def complex(a, b):
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)
    
    global_size = a.shape
    local_size = None

    prg2.complex(queue, global_size, local_size, a_buf, b_buf, dest_buf)
    c = numpy.empty_like(a)
    cl.enqueue_read_buffer(queue, dest_buf, c).wait()
    return c







#Ignore this for now
#Tiny bit more efficient, but if you screw up the global/local worksize you can make it slower
prg3 = cl.Program(ctx, """
    __kernel void sum(__global const float *a,
                      __global const float *b, 
                      __global float *c,
                      int size
                      )
    {
        int gid = get_global_id(0); 
        int gsz = get_global_size(0);
        int n = size / gsz;
        int leftover = size - n*gsz;

        for(int i = 0; i < n; i++)
        {
            c[i] = a[i] + b[i];
        }
        if(get_local_id(0) == 0)
        {
            for(int i = n*gsz; i < size; i++)
            {
                c[i] = a[i] + b[i];
            }
        }
    }
    """).build()

@timing.timings()
def add_cl2(a, b):
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)
    
    global_size = (8192,)
    local_size = (512,)
    size = numpy.int32(a.shape[0]) #size of the 1d array
    kernel_args = (a_buf,
                   b_buf,
                   dest_buf,
                   size)
    prg3.sum(queue, global_size, local_size, *(kernel_args))
    c = numpy.empty_like(a)
    cl.enqueue_read_buffer(queue, dest_buf, c).wait()
    return c
