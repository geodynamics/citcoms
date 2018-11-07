# private Pyrex/C interface ~ prefer "cimport mpi" to "cimport _mpi"


cimport cmpi


cdef class MPI_Comm:

    cdef cmpi.MPI_Comm comm
    cdef int permanent


cdef class MPI_Group:

    cdef cmpi.MPI_Group group
    cdef int permanent


# end of file
