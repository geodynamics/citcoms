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


void CoarseGridExchanger::interpretate() {
    std::cout << "in CoarseGridExchanger::interpretate" << std::endl;
}

void CoarseGridExchanger::interpolate() {
  std::cout << "in CoarseGridExchanger::interpolate" << std::endl;
  
  int n1,n2,bnid,node;
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
	      << "  receiver = "<< remoteLeader << std::endl;
    int size,nproc,lrank;
    int tag = 0;
    
    if (rank == localLeader) {
	MPI_Status status;

 	MPI_Recv(&size, 1, MPI_INT,
 		 remoteLeader, tag, intercomm, &status);
	tag ++;

	boundary = new Boundary(size);

  	MPI_Recv(boundary->bid2gid, size, MPI_INT,
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
// 	std::cout << "in CoarseGridExchanger::receiveBoundary" << std::endl;
// 	std::cout << "Grid Bounds transferred to Coarse Grid" << std::endl;
// 	std::cout << "theta= " << boundary->theta_min<< "   " << boundary->theta_max << std::endl;
// 	std::cout << "fi   = " << boundary->fi_min << "   " << boundary->fi_max << std::endl;
// 	std::cout << "r    = " << boundary->ri << "   " << boundary->ro  << std::endl;

	
	//boundary->printX();
    }
    
    // Test: Broadcast info received by localLeader to the other procs 
    // in the Coarse communicator.
    MPI_Comm_size(comm,&nproc);
    MPI_Comm_rank(comm,&lrank);
    MPI_Bcast(&size,1,MPI_INT,nproc-1,comm);

    
    if(lrank != localLeader)
      boundary = new Boundary(size);

    MPI_Bcast(boundary->bid2gid,size,MPI_INT,nproc-1,comm);
    for (int i=0; i<boundary->dim; i++) {
      MPI_Bcast(boundary->X[i],size,MPI_DOUBLE,nproc-1,comm);
    }
    MPI_Bcast(&boundary->theta_max,1,MPI_DOUBLE,nproc-1,comm);
    MPI_Bcast(&boundary->theta_min,1,MPI_DOUBLE,nproc-1,comm);
    MPI_Bcast(&boundary->fi_max,1,MPI_DOUBLE,nproc-1,comm);
    MPI_Bcast(&boundary->fi_min,1,MPI_DOUBLE,nproc-1,comm);
    MPI_Bcast(&boundary->ro,1,MPI_DOUBLE,nproc-1,comm);
    MPI_Bcast(&boundary->ri,1,MPI_DOUBLE,nproc-1,comm);

    MPI_Barrier(comm);
    std::cout << "in CoarseGridExchanger::receiveBoundary: Done!!" << std::endl;
    // Test end

    return;
}

void CoarseGridExchanger::getBid2crseelem() {
    std::cout << "in CoarseGridExchanger::getBid2crseelem" << std::endl;

    boundary->getBid2crseelem(E);

    return;
}


void CoarseGridExchanger::mapBoundary() {
    std::cout << "in CoarseGridExchanger::mapBoundary" << std::endl;
    char fname[255];
    int lrank;


    boundary->mapCoarseGrid(E, localLeader);

    // Test    
//     MPI_Comm_rank(comm,&lrank);

//     sprintf(fname,"bid2crs%d.dat",lrank);
//     ofstream file(fname);
//     for(int i=0; i< boundary->size; i++)
//       {
// 	file << "i = " << i << "elem = " << boundary->bid2elem[i] << " " << "capid = " << boundary->bid2proc[i] << endl;
//       }
//     file.close();

    return;
}



// version
// $Id: CoarseGridExchanger.cc,v 1.20 2003/09/26 18:24:58 puru Exp $

// End of file
