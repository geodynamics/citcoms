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

    MPI_Comm_rank(comm,&me);
    MPI_Comm_size(comm,&nproc);

    std::cout << "me= " << me << " nproc=" << nproc << std::endl;

    if(nproc>1) {
      if(me>0)
	local_sendVelocities();
      if(me==0)
	local_receiveVelocities();
      MPI_Barrier(intercomm);
    }
    else
      std::cout << "Don't need to run gather since nproc is " << nproc << std::endl;

    return;
}



void CoarseGridExchanger::distribute() {
    std::cout << "in CoarseGridExchanger::distribute" << std::endl;

    int me,nproc;

    MPI_Comm_rank(comm,&me);
    MPI_Comm_size(comm,&nproc);

    std::cout << "me= " << me << " nproc=" << nproc << std::endl;

    if(nproc>1) {
      if(me>0)
	local_sendVelocities();
      if(me==0)
	local_receiveVelocities();
      MPI_Barrier(intercomm);
    }
    else {
      std::cout << "in CoarseGridExchanger::distribute" << std::endl;
      std::cout << "Don't need to run gather since nproc is " << nproc << std::endl;
    }
    return;
}



void CoarseGridExchanger::interpretate() {
    std::cout << "in CoarseGridExchanger::interpretate" << std::endl;
}




void CoarseGridExchanger::impose_bc() {
    std::cout << "in CoarseGridExchanger::impose_bc" << std::endl;

}


void CoarseGridExchanger::receiveBoundary() {
    std::cout << "in CoarseGridExchanger::receiveBoundary"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader << std::endl;

    if (rank == localLeader) {
	int tag = 0;
	MPI_Status status;
	int size;

 	MPI_Recv(&size, 1, MPI_INT,
 		 remoteLeader, tag, intercomm, &status);
	tag ++;

	boundary = new Boundary(size);

  	MPI_Recv(boundary->connectivity, size, MPI_INT,
  		 remoteLeader, tag, intercomm, &status);
  	//boundary->printConnectivity();
 	tag ++;

	for (int i=0; i<boundary->dim; i++) {
  	    MPI_Recv(boundary->X[i], size, MPI_DOUBLE,
  		     remoteLeader, tag, intercomm, &status);
	    tag ++;
	}
 	MPI_Recv(&boundary->theta_max, 1, MPI_DOUBLE,
 		 remoteLeader, tag, intercomm, &status);
 	tag ++;
 	MPI_Recv(&boundary->theta_min, 1, MPI_DOUBLE,
 		 remoteLeader, tag, intercomm, &status);
 	tag ++;
 	MPI_Recv(&boundary->fi_max, 1, MPI_DOUBLE,
 		 remoteLeader, tag, intercomm, &status);
 	tag ++;
 	MPI_Recv(&boundary->fi_min, 1, MPI_DOUBLE,
 		 remoteLeader, tag, intercomm, &status);
 	tag ++;
 	MPI_Recv(&boundary->ro, 1, MPI_DOUBLE,
 		 remoteLeader, tag, intercomm, &status);
 	tag ++;
 	MPI_Recv(&boundary->ri, 1, MPI_DOUBLE,
 		 remoteLeader, tag, intercomm, &status);
 	tag ++;

	// test 
	std::cout << "in CoarseGridExchanger::receiveBoundary" << std::endl;
	std::cout << "Grid Bounds transferred to Coarse Grid" << std::endl;
	std::cout << "theta= " << boundary->theta_min<< "   " << boundary->theta_max << std::endl;
	std::cout << "fi   = " << boundary->fi_min << "   " << boundary->fi_max << std::endl;
	std::cout << "r    = " << boundary->ri << "   " << boundary->ro  << std::endl;

	
	//boundary->printX();
    }

}



void CoarseGridExchanger::mapBoundary() {
    std::cout << "in CoarseGridExchanger::mapBoundary" << std::endl;
    boundary->mapCoarseGrid(E, localLeader);
}



// version
// $Id: CoarseGridExchanger.cc,v 1.11 2003/09/19 06:32:42 ces74 Exp $

// End of file
