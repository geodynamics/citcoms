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



void FineGridExchanger::gather() {
    std::cout << "in FineGridExchanger::gather" << std::endl;

    int me,nproc;

    MPI_Comm_rank(comm,&me);
    MPI_Comm_size(comm,&nproc);

    std::cout << "me= " << me << " nproc=" << nproc << std::endl;

    if(nproc>1) {
      if(me>0)
	inter_sendVelocities();
      if(me==0)
	inter_receiveVelocities();
      MPI_Barrier(intercomm);
    }
    else 
      std::cout << "Don't need to run gather since nproc is " << nproc << std::endl;

    return;

}



void FineGridExchanger::distribute() {
    std::cout << "in FineGridExchanger::distribute" << std::endl;

    int me,nproc;

    MPI_Comm_rank(comm,&me);
    MPI_Comm_size(comm,&nproc);

    std::cout << "me= " << me << " nproc=" << nproc << std::endl;

    if(nproc>1) {
      if(me>0)
	inter_sendVelocities();
      if(me==0)
	inter_receiveVelocities();
      MPI_Barrier(intercomm);
    }
    else 
      std::cout << "Don't need to run distribute since nproc is " << nproc << std::endl;

    return;
}



void FineGridExchanger::interpretate() {
    std::cout << "in FineGridExchanger::interpretate" << std::endl;
}




void FineGridExchanger::impose_bc() {
    std::cout << "in FineGridExchanger::impose_bc" << std::endl;

}


void FineGridExchanger::createBoundary() {
    std::cout << "in FineGridExchanger::createBoundary" << std::endl;

    if (rank == localLeader) {
      // Face nodes + Edge nodes + vertex nodes
	const int size = 2*((E->mesh.nox-2)*(E->mesh.noy-2)+(E->mesh.noy-2)*(E->mesh.noz-2)+(E->mesh.noz-2)*(E->mesh.nox-2))+4*(E->mesh.nox+E->mesh.noy+E->mesh.noz-6)+8;

	boundary = new Boundary(size);

	// initialize...
	boundary->init(E);

	//boundary->printConnectivity();
	//boundary->printX();
    }
}


int FineGridExchanger::sendBoundary() {
    std::cout << "in FineGridExchanger::sendBoundary"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader << std::endl;

    if (rank == localLeader) {
	int tag = 0;
	int size = boundary->size;

	MPI_Send(&size, 1, MPI_INT,
		 remoteLeader, tag, intercomm);
	tag ++;

 	MPI_Send(boundary->connectivity, size, MPI_INT,
 		 remoteLeader, tag, intercomm);
 	tag ++;

	for (int i=0; i<boundary->dim; i++) {
	    MPI_Send(boundary->X[i], size, MPI_DOUBLE,
		     remoteLeader, tag, intercomm);
	    tag ++;
	}
    }

    return 0;
}


void FineGridExchanger::mapBoundary() {
    std::cout << "in FineGridExchanger::mapBoundary" << std::endl;
    boundary->map(E, localLeader);
}


// version
// $Id: FineGridExchanger.cc,v 1.11 2003/09/17 23:15:59 ces74 Exp $

// End of file
