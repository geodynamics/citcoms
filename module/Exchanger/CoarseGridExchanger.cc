// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>
#include <fstream>
#include <stdio.h>

#include "Boundary.h"
#include "CoarseGridExchanger.h"
#include "global_defs.h"

using std::auto_ptr;


CoarseGridExchanger::CoarseGridExchanger(const MPI_Comm comm,
					 const MPI_Comm intercomm,
					 const int leader,
					 const int localLeader,
					 const int remoteLeader,
					 const All_variables *E):
    Exchanger(comm, intercomm, leader, localLeader, remoteLeader, E)
{
    std::cout << "in CoarseGridExchanger::CoarseGridExchanger" << std::endl;
}

CoarseGridExchanger::~CoarseGridExchanger() {
    std::cout << "in CoarseGridExchanger::~CoarseGridExchanger" << std::endl;
}


void CoarseGridExchanger::gather() {
    std::cout << "in CoarseGridExchanger::gather" << std::endl;
    
    interpretate();

    if (rank == leader) {
	int nproc;
	MPI_Comm_size(comm, &nproc);

	int size = boundary->size;
	auto_ptr<double> tmp = auto_ptr<double>(new double[size]);
	double *ptmp = tmp.get();

	for (int i=0; i<nproc; i++) {
	    if (i == leader) continue; // skip leader itself

	    for (int j=0; j<boundary->dim; j++) {
		MPI_Status status;
		MPI_Recv(ptmp, size, MPI_DOUBLE,
			 i, i, comm, &status);
		for (int n=0; n<size; n++) 
		    if (boundary->bid2proc[n] == i) outgoing.v[j][n] = ptmp[n];
	    }
	}
// 	for (int n=0; n<size; n++) {
// 	    std::cout << n << " : ";
// 	    for (int j=0; j<boundary->dim; j++) 
// 		std::cout << outgoing.v[j][n] << "  ";
// 	    std::cout << std::endl;
// 	}
    }
    else {
	for (int j=0; j<boundary->dim; j++) {
	    MPI_Send(outgoing.v[j], boundary->size, MPI_DOUBLE,
		     leader, rank, comm);
	}
    }
}


void CoarseGridExchanger::distribute() {
    std::cout << "in CoarseGridExchanger::distribute" << std::endl;
}



void CoarseGridExchanger::interpretate() {
    std::cout << "in CoarseGridExchanger::interpretate" << std::endl;
    // interpolate velocity field to boundary nodes

    for(int i=0; i<boundary->size; i++) {
	int n1 = boundary->bid2elem[i];
	outgoing.v[0][i] = outgoing.v[1][i] = outgoing.v[2][i] = 0;

	if(n1 != 0) { 
	    for(int mm=1; mm<=E->sphere.caps_per_proc; mm++)
		for(int k=0; k<8; k++) {
		    int node = E->IEN[E->mesh.levmax][mm][n1].node[k+1];
		    outgoing.v[0][i] += boundary->shape[i*8+k] * E->V[mm][1][node];
		    outgoing.v[1][i] += boundary->shape[i*8+k] * E->V[mm][2][node];
		    outgoing.v[2][i] += boundary->shape[i*8+k] * E->V[mm][3][node];
		}
	}
    }
}


void CoarseGridExchanger::interpolateTemperature() {
  std::cout << "in CoarseGridExchanger::interpolateTemperature" << std::endl;
  
  int n1,n2,node;
  for(int i=0;i<boundary->size;i++)
    {
      n1=boundary->bid2elem[i];
      n2=boundary->bid2proc[i];	
//       cout << "in CoarseGridExchanger::interpolateTemperature"
// 	   << " me = " << E->parallel.me << " n1 = " << n1 << " n2 = " << n2 << endl;
      
      outgoing.T[i]=0.0;
      if(n1!=0) { 
	for(int mm=1;mm<=E->sphere.caps_per_proc;mm++)
	  for(int k=0; k< 8 ;k++)
	    {
	      node=E->IEN[E->mesh.levmax][mm][n1].node[k+1];
	      outgoing.T[i]+=boundary->shape[k]*E->T[mm][node];
	    }
	//test
// 	cout << "Interpolated...: i = " << i << " " << E->parallel.me << " " 
// 	     << outgoing.T[i] << endl;
      }
    }
  
  return;
}


void CoarseGridExchanger::impose_bc() {
    std::cout << "in CoarseGridExchanger::impose_bc" << std::endl;

}


void CoarseGridExchanger::receiveBoundary() {
    std::cout << "in CoarseGridExchanger::receiveBoundary"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  sender = "<< remoteLeader << std::endl;
    int size;
    
    if (rank == leader) {
	int tag = 0;
	MPI_Status status;

 	MPI_Recv(&size, 1, MPI_INT,
 		 remoteLeader, tag, intercomm, &status);

	boundary = new Boundary(size);
	boundary->receive(intercomm, remoteLeader);
    }

    // Broadcast info received by localLeader to the other procs 
    // in the Coarse communicator.
    MPI_Bcast(&size, 1, MPI_INT, leader, comm);

    if (rank != leader)
	boundary = new Boundary(size);

    boundary->broadcast(comm, leader);
}


void CoarseGridExchanger::mapBoundary() {
    std::cout << "in CoarseGridExchanger::mapBoundary" << std::endl;
    boundary->mapCoarseGrid(E, rank);
    boundary->sendBid2proc(comm, rank, leader);
}



// version
// $Id: CoarseGridExchanger.cc,v 1.24 2003/09/28 00:11:03 tan2 Exp $

// End of file
