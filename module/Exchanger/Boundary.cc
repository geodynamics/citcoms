// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <vector>
#include "global_defs.h"
#include "journal/journal.h"
#include "Boundary.h"


Boundary::Boundary() :
    Exchanger::Boundary()
{}


Boundary::Boundary(const All_variables* E) :
    Exchanger::Boundary()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    // boundary = all - interior
    int maxNodes = E->lmesh.nno - (E->lmesh.nox-2)
	                         * (E->lmesh.noy-2)
	                         * (E->lmesh.noz-2);
    X_.reserve(maxNodes);
    nodeID_.reserve(maxNodes);
    normal_.reserve(maxNodes);

    initX(E);
    initBBox(E);
    bbox_.print("CitcomS-Boundary-BBox");

    X_.shrink();
    X_.print("CitcomS-Boundary-X");
    nodeID_.shrink();
    nodeID_.print("CitcomS-Boundary-nodeID");
    normal_.shrink();
    normal_.print("CitcomS-Boundary-normal");
}


Boundary::~Boundary()
{}


// private functions

void Boundary::initBBox(const All_variables *E)
{
    bbox_ = tightBBox();
}


void Boundary::initX(const All_variables* E)
{
    std::vector<double> x(Exchanger::DIM);
    const int m = 1;

    for(int k=1; k<=E->lmesh.noy; k++)
	for(int j=1; j<=E->lmesh.nox; j++)
	    for(int i=1; i<=E->lmesh.noz; i++) {

		bool isBoundary = false;
		std::vector<int> normalFlag(Exchanger::DIM,0);

		if((E->parallel.me_loc[1] == 0) && (j == 1)) {
		    isBoundary |= true;
		    normalFlag[0] = -1;
		}

		if((E->parallel.me_loc[1] == E->parallel.nprocx - 1)
		   && (j == E->lmesh.nox)) {
		    isBoundary |= true;
		    normalFlag[0] = 1;
		}

		if((E->parallel.me_loc[2] == 0) && (k == 1)) {
		    isBoundary |= true;
		    normalFlag[1] = -1;
		}

		if((E->parallel.me_loc[2] == E->parallel.nprocy - 1)
		   && (k == E->lmesh.noy)) {
		    isBoundary |= true;
		    normalFlag[1] = 1;
		}

		if((E->parallel.me_loc[3] == 0) && (i == 1)) {
		    isBoundary |= true;
		    normalFlag[2] = -1;
		}

		if((E->parallel.me_loc[3] == E->parallel.nprocz - 1)
		   && (i == E->lmesh.noz)) {
		    isBoundary |= true;
		    normalFlag[2] = 1;
		}


		if(isBoundary) {
		    int node = i + (j-1)*E->lmesh.noz
			      + (k-1)*E->lmesh.noz*E->lmesh.nox;

		    for(int d=0; d<Exchanger::DIM; d++)
			x[d] = E->sx[m][d+1][node];

		    X_.push_back(x);
		    nodeID_.push_back(node);
		    normal_.push_back(normalFlag);
		}
	    }
}


// version
// $Id: Boundary.cc,v 1.52 2004/05/11 07:55:30 tan2 Exp $

// End of file
