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

#include "Array2D.h"
#include "Array2D.cc"
#include "Boundary.h"
#include "Mapping.h"
#include "CoarseGridExchanger.h"
#include "global_defs.h"


CoarseGridExchanger::CoarseGridExchanger(const MPI_Comm comm,
					 const MPI_Comm intercomm,
					 const int leader,
					 const int localLeader,
					 const int remoteLeader,
					 const All_variables *E):
    Exchanger(comm, intercomm, leader, localLeader, remoteLeader, E),
    cgmapping(NULL)
{
    std::cout << "in CoarseGridExchanger::CoarseGridExchanger" << std::endl;
}

CoarseGridExchanger::~CoarseGridExchanger() {
    std::cout << "in CoarseGridExchanger::~CoarseGridExchanger" << std::endl;
    delete cgmapping;
}


void CoarseGridExchanger::gather() {
    std::cout << "in CoarseGridExchanger::gather" << std::endl;

    interpretate();

    if (rank != leader) {
	localV.send(comm, leader);
	return;
    }

    Velo recV(boundary->size());

    int nproc;
    MPI_Comm_size(comm, &nproc);
    for (int i=0; i<nproc; i++) {
	if (i != leader) {
	    recV.receive(comm, i);
	    gatherToOutgoingV(recV, i);
	}
	else
	    gatherToOutgoingV(localV, i);
    }

    //outgoingV.print("outgoingV");
}


void CoarseGridExchanger::distribute() {
    std::cout << "in CoarseGridExchanger::distribute" << std::endl;
}



void CoarseGridExchanger::interpretate() {
    std::cout << "in CoarseGridExchanger::interpretate" << std::endl;

    // interpolate velocity field to boundary nodes
    const int size = cgmapping->size();

    for(int i=0; i<size; i++) {
	int n1 = cgmapping->bid2elem(i);
	for(int d=0; d<dim; d++)
	    localV[d][i] = 0;

	if(n1 != 0) {
	    for(int mm=1; mm<=E->sphere.caps_per_proc; mm++)
		for(int k=0; k<8; k++) {
		    int node = E->IEN[E->mesh.levmax][mm][n1].node[k+1];
		    for(int d=0; d<dim; d++)
			localV[d][i] += cgmapping->shape(i*8+k)
			              * E->sphere.cap[mm].V[d+1][node];
		}
	}
    }
    //localV.print("localV");
}


void CoarseGridExchanger::mapBoundary() {
    std::cout << "in CoarseGridExchanger::mapBoundary" << std::endl;

    createMapping();
    createDataArrays();
}


void CoarseGridExchanger::createMapping() {
    cgmapping = new CoarseGridMapping(boundary, E, comm, rank, leader);
    mapping = cgmapping;
}


void CoarseGridExchanger::createDataArrays() {
    std::cout << "in CoarseGridExchanger::createDataArrays" << std::endl;

    localV.resize(cgmapping->size());
    if (rank == leader)
	outgoingV.resize(boundary->size());
}


void CoarseGridExchanger::receiveBoundary() {
    std::cout << "in CoarseGridExchanger::receiveBoundary"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  sender = "<< remoteLeader << std::endl;

    boundary = new Boundary;
    if (rank == leader)
	boundary->receive(intercomm, remoteLeader);

    // Broadcast info received by localLeader to the other procs
    // in the Coarse communicator.
    boundary->broadcast(comm, leader);
}


void CoarseGridExchanger::interpolateTemperature() {
  std::cout << "in CoarseGridExchanger::interpolateTemperature" << std::endl;

  int n1,n2,node;
  for(int i=0;i<cgmapping->size();i++) {
      n1 = cgmapping->bid2elem(i);
      n2 = cgmapping->bid2proc(i);

      //outgoing.T[i] = 0;
      if(n1!=0) {
	for(int mm=1;mm<=E->sphere.caps_per_proc;mm++)
	  for(int k=0; k< 8 ;k++)
	    {
	      node=E->IEN[E->mesh.levmax][mm][n1].node[k+1];
	      //outgoing.T[i]+=boundary->shape[k]*E->T[mm][node];
	    }
      }
    }
}


void CoarseGridExchanger::gatherToOutgoingV(Velo& V, int sender) {
    for (int n=0; n<cgmapping->size(); n++)
	if (cgmapping->bid2proc(n) == sender)
	    for (int d=0; d<dim; d++)
		outgoingV[d][n] = V[d][n];
}


// version
// $Id: CoarseGridExchanger.cc,v 1.32 2003/10/20 17:13:08 tan2 Exp $

// End of file
