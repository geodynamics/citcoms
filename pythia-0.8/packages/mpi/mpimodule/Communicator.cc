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


// factory
mpi::Communicator * mpi::Communicator::communicator(const Group & group) const {
    MPI_Comm oldHandle = _communicator;
    MPI_Group groupHandle = group.handle();
        
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
        
    journal::debug_t info("mpi.init");
    info
        << journal::at(__HERE__)
        << "[" << rank << ":" << size << "] "
        << "creating communicator: "
        << "old=" << oldHandle << ", group=" << groupHandle
        << journal::endl;

    MPI_Comm comm;
    int status = MPI_Comm_create(oldHandle, groupHandle, &comm);
    if (status != MPI_SUCCESS) {
        journal::error_t error("mpi.init");

        error
            << journal::at(__HERE__)
            << "[" << rank << ":" << size << "] "
            << "mpi failure " << status << " while creating communicator: "
            << "old=" << oldHandle << ", group=" << groupHandle
            << journal::endl;
        return 0;
    }

    if (comm ==  MPI_COMM_NULL) {
        return 0;
    }

    return new Communicator(comm);
}


mpi::Communicator * 
mpi::Communicator::cartesian(int size, int * procs, int * periods, int reorder) const {
    MPI_Comm cartesian;
    int status = MPI_Cart_create(_communicator, size, procs, periods, reorder, &cartesian);
    if (status != MPI_SUCCESS) {
        journal::error_t error("mpi.cartesian");
        error 
            << journal::at(__HERE__)
            << "MPI_Comm_create: error " << status
            << journal::endl;
        return 0;
    }

    if (cartesian == MPI_COMM_NULL) {
        journal::error_t error("mpi.cartesian");
        error 
            << journal::at(__HERE__)
            << "MPI_Comm_create: error: null cartesian communicator"
            << journal::endl;
        return 0;
    }

    return new Communicator(cartesian);
}


// interface

int mpi::Communicator::size() const {
    int size;
    int status = MPI_Comm_size(_communicator, &size);
    if (status != MPI_SUCCESS) {
        return -1;
        }

    return size;
}


int mpi::Communicator::rank() const {
    int rank;
    int status = MPI_Comm_rank(_communicator, &rank);
    if (status != MPI_SUCCESS) {
        journal::error_t error("mpi.cartesian");
        error 
            << journal::at(__HERE__)
            << "MPI_Comm_rank: error " << status
            << journal::endl;
        return -1;
    }

    return rank;
}


void mpi::Communicator::barrier() const
{
    int status = MPI_Barrier(_communicator);
    if (status != MPI_SUCCESS) {
        journal::error_t error("mpi.cartesian");
        error 
            << journal::at(__HERE__)
            << "MPI_Barrier: error " << status
            << journal::endl;

        return;
    }

    return;
}


MPI_Comm mpi::Communicator::handle() const
{
    return _communicator;
}


void mpi::Communicator::cartesianCoordinates(int rank, int dim, int * coordinates) const {
    int status = MPI_Cart_coords(_communicator, rank, dim, coordinates);
    if (status != MPI_SUCCESS) {
        journal::error_t error("mpi.cartesian");
        error 
            << journal::at(__HERE__)
            << "MPI_Cart_coords: error " << status
            << journal::endl;

        return;
    }

    journal::debug_t info("mpi.cartesian");
    info << journal::at(__HERE__) << "coordinates:";
    for (int i=0; i < dim; ++i) {
        info << " " << coordinates[i];
    }
    info << journal::endl;

    return;
}


// constructors
mpi::Communicator::Communicator(MPI_Comm handle):
    _communicator(handle)
{}


// destructor
mpi::Communicator::~Communicator() {
    MPI_Comm_free(&_communicator);
}

// version
// $Id: Communicator.cc,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $

// End of file
