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
#include "Interior.h"


Interior::Interior() :
    Exchanger::BoundedMesh()
{}



Interior::Interior(const Exchanger::BoundedBox& remoteBBox,
		   const All_variables* E) :
    Exchanger::BoundedMesh()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    bbox_ = remoteBBox;
    bbox_.print("CitcomS-Interior-BBox");

    X_.reserve(E->lmesh.nno);
    nodeID_.reserve(E->lmesh.nno);

    initX(E);

    X_.shrink();
    X_.print("CitcomS-Interior-X");

    nodeID_.shrink();
    nodeID_.print("CitcomS-Interior-nodeID");
}


Interior::~Interior()
{}


void Interior::initX(const All_variables* E)
{
    std::vector<double> x(Exchanger::DIM);

    for (int m=1;m<=E->sphere.caps_per_proc;m++)
        for(int i=1;i<=E->lmesh.nox;i++)
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int k=1;k<=E->lmesh.noz;k++) {
                    int node = k + (i-1)*E->lmesh.noz
			     + (j-1)*E->lmesh.nox*E->lmesh.noz;

		    for(int d=0; d<Exchanger::DIM; d++)
			x[d] = E->sx[m][d+1][node];

                    if(isInside(x, bbox_)) {
                        X_.push_back(x);
                        nodeID_.push_back(node);
                    }
                }
}


// version
// $Id: Interior.cc,v 1.13 2004/05/11 07:55:30 tan2 Exp $

// End of file
