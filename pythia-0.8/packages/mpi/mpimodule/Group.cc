// -*- C++ -*-
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
#include "journal/diagnostics.h"

// local
#include "Group.h"
#include "Communicator.h"

// interface
MPI_Group mpi::Group::handle() const {
    return _group;
}


int mpi::Group::size() const {
    int size;
    int status = MPI_Group_size(_group, &size);
    if (status != MPI_SUCCESS) {
        return -1;
    }

    return size;
}


int mpi::Group::rank() const {
    int rank;
    int status = MPI_Group_rank(_group, &rank);
    if (status != MPI_SUCCESS) {
        return -1;
    }

    return rank;
}


mpi::Group * mpi::Group::group(const mpi::Communicator & comm) {
    MPI_Comm commHandle = comm.handle();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    journal::debug_t info("mpi.init");
    info
        << journal::at(__HERE__)
        << "[" << rank << ":" << size << "] "
        << "creating communicator gourp: "
        << "communicator=" << commHandle
        << journal::endl;
        
    MPI_Group group;
    int status = MPI_Comm_group(commHandle, &group);
    if (status != MPI_SUCCESS) {
        journal::error_t error("mpi.init");
        error
            << journal::at(__HERE__)
            << "[" << rank << ":" << size << "] "
            << "mpi failure " << status << " while creating communicator group: "
            << "communicator=" << commHandle
            << journal::endl;
        return 0;
    }

    if (group == MPI_GROUP_NULL) {
        return 0;
    }

    // return
    return new Group(group);
}
    
    
mpi::Group * mpi::Group::include(int size, int ranks []) const {
    MPI_Group newGroup = MPI_GROUP_NULL;
    int status = MPI_Group_incl(_group, size, ranks, &newGroup);

    if (status != MPI_SUCCESS) {
        return 0;
    }

    if (newGroup == MPI_GROUP_NULL) {
        return 0;
    }

    return new Group(newGroup);
}
    

mpi::Group * mpi::Group::exclude(int size, int ranks []) const {
    MPI_Group newGroup = MPI_GROUP_NULL;
    int status = MPI_Group_excl(_group, size, ranks, &newGroup);

    if (status != MPI_SUCCESS) {
        return 0;
    }

    if (newGroup == MPI_GROUP_NULL) {
        return 0;
    }

    return new Group(newGroup);
}


// constructor
mpi::Group::Group(MPI_Group handle):
    _group(handle)
{}


// destructor
mpi::Group::~Group() {
    MPI_Group_free(&_group);
}

// version
// $Id: Group.cc,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $
    
// End of file
    
