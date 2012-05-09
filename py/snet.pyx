cimport csnet
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

def test():
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] states = \
            np.ones((2,), dtype=np.int32)
    #cdef np.ndarray[np.int64_t, ndim=2, mode="c"] traindata = \
            #np.ones((2,2), dtype=np.int64)
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] traindata = \
            np.ones((2,2), dtype=np.int32)

    cdef unsigned int rows = <unsigned int>traindata.shape[0]
    cdef unsigned int cols = <unsigned int>traindata.shape[1]

    # Define a C-level variable to act as a 'matrix', and allocate some memory
    # for it.
    cdef int **data
    data = <int**> malloc(rows*sizeof(int *))

    # Go through the rows and pull out each one as an array of ints which can
    # be stored in our C-level 'matrix'.
    cdef unsigned int i
    for i in range(rows):
        data[i] = &(<int *>traindata.data)[i * cols]

    csnet.test2(data)
    #csnet.test2(<int**>&traindata[0,0])
    print("Hello")
    #print(data)
    x = csnet.initSimParams(2, 2, <int*>states.data, data)

#cdef convert( numpy.ndarray[numpy.int32_t, ndim = 2] incomingMatrix ):
    ## Get the size of the matrix.
    #cdef unsigned int rows = <unsigned int>incomingMatrix.shape[0]
    #cdef unsigned int cols = <unsigned int>incomingMatrix.shape[1]

    ## Define a C-level variable to act as a 'matrix', and allocate some memory
    ## for it.
    #cdef float **data
    #data = <float **>malloc(rows*sizeof(int *))

    ## Go through the rows and pull out each one as an array of ints which can
    ## be stored in our C-level 'matrix'.
    #cdef unsigned int i
    #for i in range(rows):
        #data[i] = &(<int *>incomingMatrix.data)[i * cols]

    ## Call the C function to calculate the result.
    ##cdef float result
    ##result= cSumMatrix(rows, cols, data)

    ## Free the memory we allocated for the C-level 'matrix', and we are done.
    ##free(data)
    #return data
