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
using namespace std;

#include "Boundary.h"
#include "CoarseGridExchanger.h"
#include "global_defs.h"

CoarseGridExchanger::CoarseGridExchanger(const MPI_Comm comm,
					 const MPI_Comm intercomm,
					 const int leaderRank,
					 const int localLeader,
					 const int remoteLeader,
					 const All_variables *E):
    Exchanger(comm, intercomm, leaderRank, localLeader, remoteLeader, E)
{
    std::cout << "in CoarseGridExchanger::CoarseGridExchanger" << std::endl;
}

CoarseGridExchanger::~CoarseGridExchanger() {
    std::cout << "in CoarseGridExchanger::~CoarseGridExchanger" << std::endl;
}


void CoarseGridExchanger::interpretate() {
    std::cout << "in CoarseGridExchanger::interpretate" << std::endl;
}


void CoarseGridExchanger::interpolate() {
  std::cout << "in CoarseGridExchanger::interpolate" << std::endl;
  
  int n1,n2,node;
  for(int i=0;i<boundary->size;i++)
    {
      n1=boundary->bid2elem[i];
      n2=boundary->bid2proc[i];	
      cout << "in CoarseGridExchanger::interpolate"
	   << " me = " << E->parallel.me << " n1 = " << n1 << " n2 = " << n2 << endl;
      
      outgoing.T[i]=outgoing.v[0][i]=outgoing.v[1][i]=outgoing.v[2][i]=0.0;
      if(n1!=0) { 
	for(int mm=1;mm<=E->sphere.caps_per_proc;mm++)
	  for(int k=0; k< 8 ;k++)
	    {
	      node=E->IEN[E->mesh.levmax][mm][n1].node[k+1];
	      //	      cout << "Interpolated...: node = " << node 
	      //	   << " " << E->T[mm][node]
	      //	   << endl;
	      outgoing.T[i]+=boundary->shape[i*8+k]*E->T[mm][node];
	      outgoing.v[0][i]+=boundary->shape[i*8+k]*E->V[mm][1][node];
	      outgoing.v[1][i]+=boundary->shape[i*8+k]*E->V[mm][2][node];
	      outgoing.v[2][i]+=boundary->shape[i*8+k]*E->V[mm][3][node];
	    }
	//test
	cout << "Interpolated...: i = " << i << " " << E->parallel.me << " " 
	     << outgoing.T[i] << " "
	     << outgoing.v[0][i] << " " << outgoing.v[0][i] << " " 
	     << outgoing.v[0][i] << endl;
      }
    }
  // Test
  //     std::cout << "in CoarseGridExchanger::interpolated fields" << std::endl;
  //     for(int i=0;i<boundary->size;i++)
  //       {
  // 	std::cout << i << " " << outgoing.T[i] << std::endl;
  //       }
  
  return;
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
    
    if (rank == localLeader) {
	int tag = 0;
	MPI_Status status;

 	MPI_Recv(&size, 1, MPI_INT,
 		 remoteLeader, tag, intercomm, &status);

	boundary = new Boundary(size);
	boundary->receive(intercomm, remoteLeader);
    }

    // Broadcast info received by localLeader to the other procs 
    // in the Coarse communicator.
    MPI_Bcast(&size, 1, MPI_INT, localLeader, comm);

    if (rank != localLeader)
	boundary = new Boundary(size);

    boundary->broadcast(comm, localLeader);
}


void CoarseGridExchanger::mapBoundary() {
    std::cout << "in CoarseGridExchanger::mapBoundary" << std::endl;
    boundary->mapCoarseGrid(E, lrank);
    boundary->sendBid2proc(comm, lrank, leaderRank);
}



// version
// $Id: CoarseGridExchanger.cc,v 1.23 2003/09/27 20:30:55 tan2 Exp $

// End of file
