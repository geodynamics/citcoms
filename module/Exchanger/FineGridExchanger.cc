// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>

#include "AreaWeightedNormal.h"
#include "Array2D.h"
#include "Array2D.cc"
#include "Boundary.h"
#include "Mapping.h"
#include "FineGridExchanger.h"
#include "global_defs.h"

extern "C" {
    void check_bc_consistency(const All_variables *E);
    void construct_id(const All_variables *E);
}


FineGridExchanger::FineGridExchanger(const MPI_Comm comm,
				     const MPI_Comm intercomm,
				     const int leader,
				     const int localLeader,
				     const int remoteLeader,
				     const All_variables *E):
    Exchanger(comm, intercomm, leader, localLeader, remoteLeader, E),
    fgmapping(NULL),
    awnormal(NULL)
{
    std::cout << "in FineGridExchanger::FineGridExchanger" << std::endl;
}


FineGridExchanger::~FineGridExchanger() {
    std::cout << "in FineGridExchanger::~FineGridExchanger" << std::endl;
    delete fgmapping;
    delete awnormal;
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


void FineGridExchanger::mapBoundary() {
    std::cout << "in FineGridExchanger::mapBoundary" << std::endl;

    // Assuming all boundary nodes are inside localLeader!
    // assumption will be relaxed in future
    if (rank == leader) {
	createMapping();
	createDataArrays();
    }
}


void FineGridExchanger::createMapping() {
    fgmapping = new FineGridMapping(boundary, E, comm, rank, leader);
    mapping = fgmapping;
    awnormal = new AreaWeightedNormal(boundary, E, fgmapping);
}


void FineGridExchanger::createDataArrays() {
    std::cout << "in FineGridExchanger::createDataArrays" << std::endl;

    if (rank == leader) {
	incomingV.resize(boundary->size());
	old_incomingV.resize(boundary->size());
    }
}


void FineGridExchanger::createBoundary() {
    std::cout << "in FineGridExchanger::createBoundary" << std::endl;

    if (rank == leader) {
	boundary = new Boundary(E);
    }
}


void FineGridExchanger::sendBoundary() {
    std::cout << "in FineGridExchanger::sendBoundary"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader << std::endl;

    if (rank == leader)
	boundary->send(intercomm, remoteLeader);
}


void FineGridExchanger::setBCFlag() {
    std::cout << "in FineGridExchanger::setBCFlag" << std::endl;

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
    std::cout << "in FineGridExchanger::imposeConstraint" << std::endl;

    if(rank == leader) {
	awnormal->imposeConstraint(incomingV);
    }
}


void FineGridExchanger::imposeBC() {
    std::cout << "in FineGridExchanger::imposeBC" << std::endl;

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
//                 std::cout << E->sphere.cap[m].VB[1][n] << " "
// 			  << E->sphere.cap[m].VB[2][n] << " "
// 			  <<  E->sphere.cap[m].VB[3][n] << std::endl;
	    }
	}
    }
}


// version
// $Id: FineGridExchanger.cc,v 1.26 2003/10/20 17:13:08 tan2 Exp $

// End of file
