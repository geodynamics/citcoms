// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <algorithm>
#include <limits>
#include <vector>
#include "global_defs.h"
#include "journal/journal.h"
#include "Boundary.h"


Boundary::Boundary() :
    BoundedMesh()
{}


Boundary::Boundary(const All_variables* E) :
    BoundedMesh()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Boundary::Boundary" << journal::end;

    // boundary = all - interior
    int maxNodes = E->lmesh.nno - (E->lmesh.nox-2)
	                         *(E->lmesh.noy-2)
	                         *(E->lmesh.noz-2);
    X_.reserve(maxNodes);
    nodeID_.reserve(maxNodes);

    initX(E);
    initBBox(E);
    bbox_.print("Boundary-BBox");


    X_.shrink();
    X_.print("Boundary-X");
    nodeID_.shrink();
    nodeID_.print("Boundary-nodeID");
}


void Boundary::initBBox(const All_variables *E)
{
    bbox_ = tightBBox();
}


void Boundary::initX(const All_variables* E)
{
    std::vector<double> x(DIM);
    const int m = 1;

    for(int k=1; k<=E->lmesh.noy; k++)
	for(int j=1; j<=E->lmesh.nox; j++)
	    for(int i=1; i<=E->lmesh.noz; i++) {

		if(isOnBoundary(E, i, j, k)) {
		    int node = i + (j-1)*E->lmesh.noz
			     + (k-1)*E->lmesh.noz*E->lmesh.nox;

		    for(int d=0; d<DIM; d++)
			x[d] = E->sx[m][d+1][node];

		    X_.push_back(x);
		    nodeID_.push_back(node);
		}
	    }
}


bool Boundary::isOnBoundary(const All_variables* E, int i, int j, int k) const
{
    return ((E->parallel.me_loc[1] == 0) && (j == 1)) ||
	   ((E->parallel.me_loc[2] == 0) && (k == 1)) ||
	   ((E->parallel.me_loc[3] == 0) && (i == 1)) ||
   	   ((E->parallel.me_loc[1] == E->parallel.nprocx - 1)
	     && (j == E->lmesh.nox)) ||
	   ((E->parallel.me_loc[2] == E->parallel.nprocy - 1)
	     && (k == E->lmesh.noy)) ||
	   ((E->parallel.me_loc[3] == E->parallel.nprocz - 1)
	     && (i == E->lmesh.noz));
}


// version
// $Id: Boundary.cc,v 1.49 2004/01/14 18:52:19 tan2 Exp $

// End of file
