//
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                              Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
//
// <LicenseText>
//
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <mpi.h>

// Force MPI_* symbols to be pulled into the main executable
// (mpipython.exe), so that they will available to Python extension
// modules.  This is necessary when linking against a static MPI
// library on platforms where PIC and non-PIC code don't mix.

void mpipython_dummy() {
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Group group = MPI_GROUP_NULL;
    int rank, size, reorder, dim, count, source, dest, tag;
    int sendcount, sendtag, recvcount, recvtag, flag, root;
    int *procs, *periods, *coordinates;
    void *buf, *sendbuf, *recvbuf;
    int ranks[42];
    MPI_Op op;
    MPI_Datatype datatype, sendtype, recvtype;
    MPI_Request request, array_of_requests[42];
    MPI_Status status, array_of_statuses[42];

    rank = size = reorder = dim = count = source = dest = tag = 0;
    sendcount = sendtag = recvcount = recvtag = flag = root = 0;
    datatype = sendtype = recvtype = MPI_INT;
    op = MPI_SUM;
    procs = periods = coordinates = 0;
    buf = sendbuf = recvbuf = 0;
    
    MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    MPI_Barrier(comm);
    MPI_Cart_coords(comm, rank, dim, coordinates);
    MPI_Cart_create(comm, size, procs, periods, reorder, &comm);
    MPI_Comm_create(comm, group, &comm);
    MPI_Comm_free(&comm);
    MPI_Comm_group(comm, &group);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    MPI_Get_count(&status, datatype, &count);
    MPI_Group_excl(group, size, ranks, &group);
    MPI_Group_free(&group);
    MPI_Group_incl(group, size, ranks, &group);
    MPI_Group_rank(group, &rank);
    MPI_Group_size(group, &size);
    MPI_Initialized(&flag);
    MPI_Irecv(buf, count, datatype, source, tag, comm, &request);
    MPI_Isend(buf, count, datatype, dest, tag, comm, &request);
    MPI_Recv(buf, count, datatype, source, tag, comm, &status);
    MPI_Send(buf, count, datatype, dest, tag, comm);
    MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, &status);
    MPI_Waitall(count, array_of_requests, array_of_statuses);
    MPI_Wtime();
}

// version
// $Id: dummy.cc,v 1.2 2005/10/08 00:17:48 leif Exp $
    
// End of file
