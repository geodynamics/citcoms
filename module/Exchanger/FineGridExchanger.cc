// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>

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



void FineGridExchanger::gather() {
    std::cout << "in FineGridExchanger::gather" << std::endl;
}



void FineGridExchanger::distribute() {
    std::cout << "in FineGridExchanger::distribute" << std::endl;
}



void FineGridExchanger::interpretate() {
    std::cout << "in FineGridExchanger::interpretate" << std::endl;
}




void FineGridExchanger::impose_bc() {
    std::cout << "in FineGridExchanger::impose_bc" << std::endl;

}


const Boundary* FineGridExchanger::createBoundary() {
    std::cout << "in FineGridExchanger::createBoundary" << std::endl;

    Boundary* b = NULL;

    if (rank == localLeader) {
	const int size = 10;
	b = new Boundary(size);

	// initialize...

	// test
	int *c = b->connectivity.get();
	for(int j=0; j<size; j++)
	    c[j] = j;

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
	MPI_Send(b->connectivity.get(), size, MPI_INT,
		 remoteLeader, tag, intercomm);
	tag ++;
	for (int i=0; i<b->dim; i++, tag++) {
	    MPI_Send(b->X[i].get(), size, MPI_DOUBLE,
		     remoteLeader, tag, intercomm);
	}
    }

    return 0;
}



// version
// $Id: FineGridExchanger.cc,v 1.3 2003/09/09 02:35:22 tan2 Exp $

// End of file
