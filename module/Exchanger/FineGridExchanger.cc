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



FineGridExchanger::FineGridExchanger(MPI_Comm communicator,
				     MPI_Comm icomm,
				     int local,
				     int remote,
				     const All_variables *e):
    Exchanger(communicator, icomm, local, remote, e)
{
    std::cout << "in FineGridExchanger::FineGridExchanger" << std::endl;
}



FineGridExchanger::~FineGridExchanger() {
    std::cout << "in FineGridExchanger::~FineGridExchanger" << std::endl;
}



void FineGridExchanger::gather(const Boundary* b) {
    std::cout << "in FineGridExchanger::gather" << std::endl;
}



void FineGridExchanger::distribute(const Boundary* b) {
    std::cout << "in FineGridExchanger::distribute" << std::endl;
}



void FineGridExchanger::interpretate(const Boundary* b) {
    std::cout << "in FineGridExchanger::interpretate" << std::endl;
}




void FineGridExchanger::impose_bc(const Boundary* b) {
    std::cout << "in FineGridExchanger::impose_bc" << std::endl;

}


const Boundary* FineGridExchanger::createBoundary() {
    std::cout << "in FineGridExchanger::createBoundary" << std::endl;

    Boundary* b = NULL;

    if (rank == localLeader) {
      // Face nodes + Edge nodes + vertex nodes
	const int size = 2*((E->mesh.nox-2)*(E->mesh.noy-2)+(E->mesh.noy-2)*(E->mesh.noz-2)+(E->mesh.noz-2)*(E->mesh.nox-2))+4*(E->mesh.nox+E->mesh.noy+E->mesh.noz-6)+8;
	b = new Boundary(size);

	// initialize...
	b->init(E);

	//b->printConnectivity();
    }

    return b;
}


int FineGridExchanger::sendBoundary(const Boundary* b) {
    std::cout << "in FineGridExchanger::sendBoundary"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader << std::endl;

    if (rank == localLeader) {
	int tag = 0;
	int size = b->size;

	MPI_Send(&size, 1, MPI_INT,
		 remoteLeader, tag, intercomm);
	tag ++;
	MPI_Send(b->connectivity, size, MPI_INT,
		 remoteLeader, tag, intercomm);
	tag ++;
	for (int i=0; i<b->dim; i++, tag++) {
	    MPI_Send(b->X[i], size, MPI_DOUBLE,
		     remoteLeader, tag, intercomm);
	}
    }

    return 0;
}


void FineGridExchanger::mapBoundary(const Boundary* b) {

}


// version
// $Id: FineGridExchanger.cc,v 1.7 2003/09/09 20:41:57 puru Exp $

// End of file
