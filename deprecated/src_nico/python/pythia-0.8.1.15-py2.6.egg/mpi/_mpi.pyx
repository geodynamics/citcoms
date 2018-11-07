# Process this file with Pyrex to produce mpi.c


cdef extern from "stdlib.h":
    void *malloc(int)
    void free(void *)


cdef class MPI_Comm:

    def __init__(MPI_Comm self):
        self.comm = cmpi.MPI_COMM_NULL
        self.permanent = 0

    def __dealloc__(MPI_Comm self):
        cdef int error
        if self.permanent:
            return
        error = cmpi.MPI_Comm_free(&self.comm)
        if error != cmpi.MPI_SUCCESS:
            # Will anyone hear our cries?
            raise MPI_Error(error)
        return


cdef class MPI_Group:

    def __init__(MPI_Group self):
        self.group = cmpi.MPI_GROUP_NULL
        self.permanent = 0

    def __dealloc__(MPI_Group self):
        cdef int error
        if self.permanent:
            return
        error = cmpi.MPI_Group_free(&self.group)
        if error != cmpi.MPI_SUCCESS:
            raise MPI_Error(error)
        return


cdef permanentCommObj(cmpi.MPI_Comm comm):
    cdef MPI_Comm obj
    obj = MPI_Comm()
    obj.comm = comm
    obj.permanent = 1
    return obj

cdef permanentGroupObj(cmpi.MPI_Group group):
    cdef MPI_Group obj
    obj = MPI_Group()
    obj.group = group
    obj.permanent = 1
    return obj


MPI_COMM_WORLD  = permanentCommObj(cmpi.MPI_COMM_WORLD)
MPI_COMM_NULL   = permanentCommObj(cmpi.MPI_COMM_NULL)
MPI_COMM_SELF   = permanentCommObj(cmpi.MPI_COMM_SELF)

MPI_GROUP_NULL  = permanentGroupObj(cmpi.MPI_GROUP_NULL)
MPI_GROUP_EMPTY = permanentGroupObj(cmpi.MPI_GROUP_EMPTY)


cdef getCommObj(cmpi.MPI_Comm comm):
    cdef MPI_Comm obj
    if comm == cmpi.MPI_COMM_WORLD:
        return MPI_COMM_WORLD
    elif comm == cmpi.MPI_COMM_NULL:
        return MPI_COMM_NULL
    elif comm == cmpi.MPI_COMM_SELF:
        return MPI_COMM_SELF
    obj = MPI_Comm()
    obj.comm = comm
    return obj

cdef getGroupObj(cmpi.MPI_Group group):
    cdef MPI_Group obj
    if group == cmpi.MPI_GROUP_NULL:
        return MPI_GROUP_NULL
    elif group == cmpi.MPI_GROUP_EMPTY:
        return MPI_GROUP_EMPTY
    obj = MPI_Group()
    obj.group = group
    return obj


class MPI_Error(EnvironmentError):
    def __str__(self):
        return MPI_Error_string(self.args[0])


def MPI_Barrier(MPI_Comm comm):
    cdef int error
    error = cmpi.MPI_Barrier(comm.comm)
    if error != cmpi.MPI_SUCCESS:
        raise MPI_Error(error)
    return


def MPI_Comm_create(MPI_Comm comm, MPI_Group group):
    cdef int error
    cdef cmpi.MPI_Comm comm_out
    error = cmpi.MPI_Comm_create(comm.comm, group.group, &comm_out)
    if error != cmpi.MPI_SUCCESS:
        raise MPI_Error(error)
    return getCommObj(comm_out)


def MPI_Comm_group(MPI_Comm comm):
    cdef cmpi.MPI_Group group
    error = cmpi.MPI_Comm_group(comm.comm, &group)
    if error != cmpi.MPI_SUCCESS:
        raise MPI_Error(error)
    return getGroupObj(group)


def MPI_Comm_rank(MPI_Comm comm):
    cdef int error
    cdef int rank
    error = cmpi.MPI_Comm_rank(comm.comm, &rank)
    if error != cmpi.MPI_SUCCESS:
        raise MPI_Error(error)
    return rank


def MPI_Comm_size(MPI_Comm comm):
    cdef int error
    cdef int size
    error = cmpi.MPI_Comm_size(comm.comm, &size)
    if error != cmpi.MPI_SUCCESS:
        raise MPI_Error(error)
    return size


cdef char cstring[1024]

def MPI_Error_string(int errorcode):
    cdef int error
    cdef int resultlen
    error = cmpi.MPI_Error_string(errorcode, cstring, &resultlen)
    if error != cmpi.MPI_SUCCESS:
        raise MPI_Error(error)
    if resultlen >= 1024:
        raise RuntimeError("buffer overflow")
    string = cstring
    return string


def MPI_Group_excl(MPI_Group group, members):
    cdef int error
    cdef int n
    cdef int *ranks
    cdef cmpi.MPI_Group group_out
    
    n = len(members)
    ranks = <int *>malloc(n * sizeof(int))
    for i from 0 <= i < n:
        ranks[i] = members[i]

    error = cmpi.MPI_Group_excl(group.group, n, ranks, &group_out)
    
    free(ranks)
    
    if error != cmpi.MPI_SUCCESS:
        raise MPI_Error(error)
    
    return getGroupObj(group_out)


def MPI_Group_incl(MPI_Group group, members):
    cdef int error
    cdef int n
    cdef int *ranks
    cdef cmpi.MPI_Group group_out
    
    n = len(members)
    ranks = <int *>malloc(n * sizeof(int))
    for i from 0 <= i < n:
        ranks[i] = members[i]

    error = cmpi.MPI_Group_incl(group.group, n, ranks, &group_out)
    
    free(ranks)
    
    if error != cmpi.MPI_SUCCESS:
        raise MPI_Error(error)
    
    return getGroupObj(group_out)


def MPI_Group_rank(MPI_Group group):
    cdef int error
    cdef int rank
    error = cmpi.MPI_Group_rank(group.group, &rank)
    if error != cmpi.MPI_SUCCESS:
        raise MPI_Error(error)
    return rank


def MPI_Group_size(MPI_Group group):
    cdef int error
    cdef int size
    error = cmpi.MPI_Group_size(group.group, &size)
    if error != cmpi.MPI_SUCCESS:
        raise MPI_Error(error)
    return size


# end of file
