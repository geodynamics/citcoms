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
#include "global_defs.h"
#include "journal/journal.h"
#include "BoundedBox.h"
#include "BoundedMesh.h"
#include "VTInterpolator.h"


VTInterpolator::VTInterpolator(const BoundedMesh& boundedMesh,
			       const All_variables* E,
			       Array2D<int,1>& meshNode) :
    FEMInterpolator(boundedMesh, E, meshNode)
{}


void VTInterpolator::interpolateStress(Array2D<double,STRESS_DIM>& S)
{
    S.assign(size(), 0);

    const int mm = 1;
    for(int i=0; i<size(); i++) {
	int n1 = elem_[0][i];
	for(int k=0; k<NODES_PER_ELEMENT; k++) {
	    int node = E->ien[mm][n1].node[k+1] - 1;
	    for(int d=0; d<STRESS_DIM; d++)
		S[d][i] += shape_[k][i] * E->gstress[mm][node*STRESS_DIM+d];
	}
    }
}


void VTInterpolator::interpolateTemperature(Array2D<double,1>& T)
{
    T.assign(size(), 0);

    const int mm = 1;
    for(int i=0; i<size(); i++) {
	int n1 = elem_[0][i];
	for(int k=0; k<NODES_PER_ELEMENT; k++) {
	    int node = E->ien[mm][n1].node[k+1];
	    T[0][i] += shape_[k][i] * E->T[mm][node];
	}
    }
}


void VTInterpolator::interpolateVelocity(Array2D<double,DIM>& V)
{
    V.assign(size(), 0);

    const int mm = 1;
    for(int i=0; i<size(); i++) {
	int n1 = elem_[0][i];
	for(int k=0; k<NODES_PER_ELEMENT; k++) {
	    int node = E->ien[mm][n1].node[k+1];
	    for(int d=0; d<DIM; d++)
		V[d][i] += shape_[k][i] * E->sphere.cap[mm].V[d+1][node];
	}
    }
}


// private functions


// version
// $Id: VTInterpolator.cc,v 1.2 2004/04/14 20:06:24 tan2 Exp $

// End of file
