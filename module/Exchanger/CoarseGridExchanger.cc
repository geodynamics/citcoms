// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>

#include "Boundary.h"
#include "CoarseGridExchanger.h"



CoarseGridExchanger::CoarseGridExchanger(MPI_Comm communicator,
					 MPI_Comm icomm,
					 int local,
					 int remote,
					 const All_variables *e):
    Exchanger(communicator, icomm, local, remote, e)
{
    std::cout << "in CoarseGridExchanger::CoarseGridExchanger" << std::endl;
}



CoarseGridExchanger::~CoarseGridExchanger() {
    std::cout << "in CoarseGridExchanger::~CoarseGridExchanger" << std::endl;
}



void CoarseGridExchanger::gather() {
    std::cout << "in CoarseGridExchanger::gather" << std::endl;

    int me,nproc;
    MPI_Comm_size(intercomm,&nproc);
    MPI_Comm_rank(intercomm,&me);
    if(me>0)
      sendVelocities();
    if(me==0)
      receiveVelocities();
    MPI_Barrier(intercomm);

    return;
}



void CoarseGridExchanger::distribute() {
    std::cout << "in CoarseGridExchanger::distribute" << std::endl;
}



void CoarseGridExchanger::interpretate() {
    std::cout << "in CoarseGridExchanger::interpretate" << std::endl;
}




void CoarseGridExchanger::impose_bc() {
    std::cout << "in CoarseGridExchanger::impose_bc" << std::endl;

}


const Boundary* CoarseGridExchanger::receiveBoundary() {
    std::cout << "in CoarseGridExchanger::receiveBoundary"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader << std::endl;

    Boundary *b = NULL;

    if (rank == localLeader) {
	int tag = 0;
	MPI_Status status;
	int size;
	MPI_Recv(&size, 1, MPI_INT,
		 remoteLeader, tag, intercomm, &status);
	tag ++;

	b = new Boundary(size);

	MPI_Recv(b->connectivity, size, MPI_INT,
		 remoteLeader, tag, intercomm, &status);
	tag ++;
	for (int i=0; i<b->dim; i++, tag++) {
	    MPI_Recv(b->X[i], size, MPI_DOUBLE,
		     remoteLeader, tag, intercomm, &status);
	}

	boundary = b;
	b->printConnectivity();
    }

    return b;
}


void CoarseGridExchanger::mapBoundary(Boundary* b) {

}



// version
// $Id: CoarseGridExchanger.cc,v 1.6 2003/09/10 21:11:09 puru Exp $

// End of file
