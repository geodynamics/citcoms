// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <iostream>
#include "AreaWeightedNormal.h"
#include "Array2D.h"
#include "Boundary.h"
#include "Mapping.h"
#include "FineGridExchanger.h"
#include "global_defs.h"
#include "journal/journal.h"


extern "C" {
    void check_bc_consistency(const All_variables *E);
    void construct_id(const All_variables *E);
}


FineGridExchanger::FineGridExchanger(const MPI_Comm comm,
				     const MPI_Comm intercomm,
				     const int leader,
				     const int remoteLeader,
				     const All_variables *E):
    Exchanger(comm, intercomm, leader, remoteLeader, E),
    fgmapping(NULL),
    awnormal(NULL)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in FineGridExchanger::FineGridExchanger" << journal::end;
}


FineGridExchanger::~FineGridExchanger() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in FineGridExchanger::~FineGridExchanger" << journal::end;
    delete awnormal;
    delete fgmapping;
}


void FineGridExchanger::gather() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in FineGridExchanger::gather" << journal::end;
    interpolateTemperature();
    
}


void FineGridExchanger::distribute() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in FineGridExchanger::distribute" << journal::end;
}


void FineGridExchanger::interpretate() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in FineGridExchanger::interpretate" << journal::end;
}


void FineGridExchanger::mapBoundary() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in FineGridExchanger::mapBoundary" << journal::end;

    createMapping();
    createDataArrays();
}


void FineGridExchanger::createBoundary() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in FineGridExchanger::createBoundary" << journal::end;

    boundary = new Boundary(E);
}


void FineGridExchanger::createMapping() {
    // init boundary->X and fgmapping
    intmapping = new CoarseGridMapping(interior, E, comm, rank, leader);
    fgmapping = new FineGridMapping(boundary, E, comm, rank, leader);
    awnormal = new AreaWeightedNormal(boundary, E, fgmapping);
    mapping = fgmapping;
}


void FineGridExchanger::createDataArrays() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in FineGridExchanger::createDataArrays" << journal::end;

    localV.resize(fgmapping->size());
    if (rank == leader) {
	incomingV.resize(boundary->size());
	old_incomingV.resize(boundary->size());
    }
}


void FineGridExchanger::sendBoundary() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in FineGridExchanger::sendBoundary"
	  << "  rank = " << rank
	  << "  receiver = "<< remoteLeader << journal::end;

    if (rank == leader)
	boundary->send(intercomm, remoteLeader);
}

void FineGridExchanger::receiveInterior() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in CoarseGridExchanger::receiveBoundary"
	  << "  rank = " << rank
	  << "  sender = "<< remoteLeader << journal::end;

    boundary = new Boundary;
    if (rank == leader)
	interior->receive(intercomm, remoteLeader);

    // Broadcast info received by leader to the other procs
    // in the Coarse communicator.
    interior->broadcast(comm, leader);
}


void FineGridExchanger::setBCFlag() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in FineGridExchanger::setBCFlag" << journal::end;

    // Because CitcomS is defaulted to have reflecting side BC,
    // here we should change to velocity BC.
    for(int m=1; m<=E->sphere.caps_per_proc; m++)
	for(int i=0; i<fgmapping->size(); i++) {
	    int n = fgmapping->bid2gid(i);
	    int p = fgmapping->bid2proc(i);
	    if (p == rank) {
// 		std::cout << "    before: " << std::hex
// 			  << E->node[m][n] << std::dec << std::endl;
		E->node[m][n] = E->node[m][n] | VBX;
		E->node[m][n] = E->node[m][n] | VBY;
		E->node[m][n] = E->node[m][n] | VBZ;
		E->node[m][n] = E->node[m][n] & (~SBX);
		E->node[m][n] = E->node[m][n] & (~SBY);
		E->node[m][n] = E->node[m][n] & (~SBZ);
// 		std::cout << "    after : "  << std::hex
// 			  << E->node[m][n] << std::dec << std::endl;
	    }
	}

    check_bc_consistency(E);
    // reconstruct ID array to reflect changes in BC
    construct_id(E);
}


void FineGridExchanger::imposeConstraint(){
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in FineGridExchanger::imposeConstraint" << journal::end;

    if(rank == leader) {
	awnormal->imposeConstraint(incomingV);
    }
}


void FineGridExchanger::imposeBC() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in FineGridExchanger::imposeBC" << journal::end;

    journal::debug_t debugBC("imposeBC");
    debugBC << journal::loc(__HERE__);

    double N1, N2;

    if(cge_t == 0) {
        N1 = 0.0;
        N2 = 1.0;
    } else {
        N1 = (cge_t - fge_t) / cge_t;
        N2 = fge_t / cge_t;
    }

    for(int m=1; m<=E->sphere.caps_per_proc; m++) {
	for(int i=0; i<fgmapping->size(); i++) {
	    int n = fgmapping->bid2gid(i);
	    int p = fgmapping->bid2proc(i);
	    if (p == rank) {
 		for(int d=0; d<dim; d++)
 		    E->sphere.cap[m].VB[d+1][n] = N1 * old_incomingV[d][i]
 		                                + N2 * incomingV[d][i];
		debugBC << E->sphere.cap[m].VB[1][n] << " "
			<< E->sphere.cap[m].VB[2][n] << " "
			<<  E->sphere.cap[m].VB[3][n] << journal::newline;
	    }
	}
    }
    debugBC << journal::end;
}

void FineGridExchanger::interpolateTemperature() {
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in FineGridExchanger::interpolateTemperature" << journal::end;

  int n1,n2,node;
  for(int i=0;i<intmapping->size();i++) {
      n1 = intmapping->bid2elem(i);
      n2 = intmapping->bid2proc(i);

      CoutgoingT[0][i] = 0;
      if(n1!=0) {
	for(int mm=1;mm<=E->sphere.caps_per_proc;mm++)
	  for(int k=0; k< 8 ;k++)
	    {
	      node=E->IEN[E->mesh.levmax][mm][n1].node[k+1];
	      CoutgoingT[0][i] += intmapping->shape(k) * E->T[mm][node];
	    }
      }
    }
}

// version
// $Id: FineGridExchanger.cc,v 1.28 2003/10/28 02:34:37 puru Exp $

// End of file
