// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>

#include "global_defs.h"
#include "Boundary.h"
#include "FineGridExchanger.h"


FineGridExchanger::FineGridExchanger(const MPI_Comm comm,
				     const MPI_Comm intercomm,
				     const int leader,
				     const int localLeader,
				     const int remoteLeader,
				     const All_variables *E):
    Exchanger(comm, intercomm, leader, localLeader, remoteLeader, E)
{
    std::cout << "in FineGridExchanger::FineGridExchanger" << std::endl;
}


FineGridExchanger::~FineGridExchanger() {
    std::cout << "in FineGridExchanger::~FineGridExchanger" << std::endl;
}


void FineGridExchanger::gather() {
    std::cout << "in FineGridExchanger::gather" << std::endl;
}


void FineGridExchanger::distribute() {
    std::cout << "in FineGridExchanger::distribute" << std::endl;
}


void FineGridExchanger::interpretate() {
    std::cout << "in FineGridExchanger::interpretate" << std::endl;
}


void FineGridExchanger::createBoundary() {
    std::cout << "in FineGridExchanger::createBoundary" << std::endl;

    if (rank == leader) {
	// boundary = all - interior
	int size = E->mesh.nno - (E->mesh.nox-2)*(E->mesh.noy-2)*(E->mesh.noz-2);
	boundary = new Boundary(size);
    }
}


void FineGridExchanger::sendBoundary() {
    std::cout << "in FineGridExchanger::sendBoundary"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader << std::endl;

    if (rank == leader) {
	int tag = 0;
	int itmp = boundary->size;
	MPI_Send(&itmp, 1, MPI_INT,
		 remoteLeader, tag, intercomm);

	boundary->send(intercomm, remoteLeader);
    }
}


void FineGridExchanger::mapBoundary() {
    std::cout << "in FineGridExchanger::mapBoundary" << std::endl;

    // Assuming all boundary nodes are inside localLeader!
    // assumption will be relaxed in future
    if (rank == leader) {
	boundary->mapFineGrid(E);
	createDataArrays();
    }
}


// version
// $Id: FineGridExchanger.cc,v 1.21 2003/10/03 18:12:41 tan2 Exp $

// End of file
