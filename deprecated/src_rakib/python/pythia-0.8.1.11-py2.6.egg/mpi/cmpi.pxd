# Pyrex module for MPI

cdef extern from "mpi.h":

    enum MPI_Error:
        MPI_SUCCESS

    ctypedef struct MPI_Comm_Imp:
        pass
    ctypedef MPI_Comm_Imp *MPI_Comm
    MPI_Comm MPI_COMM_NULL
    MPI_Comm MPI_COMM_SELF
    MPI_Comm MPI_COMM_WORLD

    ctypedef struct MPI_Group_Imp:
        pass
    ctypedef MPI_Group_Imp *MPI_Group
    MPI_Group MPI_GROUP_NULL
    MPI_Group MPI_GROUP_EMPTY

    int MPI_Init(int *, char ***)
    int MPI_Finalize()

    int MPI_Barrier(MPI_Comm comm)
    
    int MPI_Comm_create(MPI_Comm, MPI_Group, MPI_Comm *)
    int MPI_Comm_free(MPI_Comm *)
    int MPI_Comm_group(MPI_Comm comm, MPI_Group *group)
    int MPI_Comm_rank(MPI_Comm, int *)
    int MPI_Comm_size(MPI_Comm, int *)
    
    int MPI_Error_string(int, char *, int *)

    int MPI_Group_excl(MPI_Group, int, int *, MPI_Group *)
    int MPI_Group_free(MPI_Group *)
    int MPI_Group_incl(MPI_Group, int, int *, MPI_Group *)
    int MPI_Group_rank(MPI_Group, int *)
    int MPI_Group_size(MPI_Group, int *)


# end of file
