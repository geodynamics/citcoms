// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "Array2D.h"
#include "Boundary.h"
#include "Mapping.h"
#include "CoarseGridExchanger.h"
#include "global_defs.h"
#include "journal/journal.h"


CoarseGridExchanger::CoarseGridExchanger(const MPI_Comm comm,
					 const MPI_Comm intercomm,
					 const int leader,
					 const int remoteLeader,
					 const All_variables *E):
    Exchanger(comm, intercomm, leader, remoteLeader, E),
    cgmapping(NULL)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in CoarseGridExchanger::CoarseGridExchanger" << journal::end;
}

CoarseGridExchanger::~CoarseGridExchanger() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in CoarseGridExchanger::~CoarseGridExchanger" << journal::end;
    delete cgmapping;
}


void CoarseGridExchanger::gather() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in CoarseGridExchanger::gather" << journal::end;

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
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in CoarseGridExchanger::distribute" << journal::end;
}



void CoarseGridExchanger::interpretate() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in CoarseGridExchanger::interpretate" << journal::end;

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
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in CoarseGridExchanger::mapBoundary" << journal::end;

    createMapping();
    createDataArrays();
}


void CoarseGridExchanger::createMapping() {
    cgmapping = new CoarseGridMapping(boundary, E, comm, rank, leader);
    mapping = cgmapping;
}


void CoarseGridExchanger::createDataArrays() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in CoarseGridExchanger::createDataArrays" << journal::end;

    localV.resize(cgmapping->size());
    if (rank == leader)
	outgoingV.resize(boundary->size());
}


void CoarseGridExchanger::receiveBoundary() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in CoarseGridExchanger::receiveBoundary"
	  << "  rank = " << rank
	  << "  sender = "<< remoteLeader << journal::end;

    boundary = new Boundary;
    if (rank == leader)
	boundary->receive(intercomm, remoteLeader);

    // Broadcast info received by leader to the other procs
    // in the Coarse communicator.
    boundary->broadcast(comm, leader);
}


void CoarseGridExchanger::interpolateTemperature() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in CoarseGridExchanger::interpolateTemperature" << journal::end;

  int n1,n2,node;
  for(int i=0;i<cgmapping->size();i++) {
      n1 = cgmapping->bid2elem(i);
      n2 = cgmapping->bid2proc(i);

      outgoingT[0][i] = 0;
      if(n1!=0) {
	for(int mm=1;mm<=E->sphere.caps_per_proc;mm++)
	  for(int k=0; k< 8 ;k++)
	    {
	      node=E->IEN[E->mesh.levmax][mm][n1].node[k+1];
	      outgoingT[0][i] += cgmapping->shape(k) * E->T[mm][node];
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
// $Id: CoarseGridExchanger.cc,v 1.33 2003/10/24 04:51:53 tan2 Exp $

// End of file
