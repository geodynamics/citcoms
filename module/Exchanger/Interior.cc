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
#include "BoundedBox.h"
#include "Interior.h"


Interior::Interior(bool dimensional) :
    BoundedMesh(dimensional)
{}



Interior::Interior(const BoundedBox& remoteBBox, const All_variables* E,
		   bool dimensional) :
    BoundedMesh(dimensional)
{
    bbox_ = remoteBBox;
    bbox_.print("Interior-BBox");

    X_.reserve(E->lmesh.nno);
    nodeID_.reserve(E->lmesh.nno);

    initX(E);

    X_.shrink();
    X_.print("Interior-X");
    nodeID_.shrink();
    nodeID_.print("Interior-nodeID");
}


void Interior::initX(const All_variables* E)
{
    std::vector<double> x(DIM);

    for (int m=1;m<=E->sphere.caps_per_proc;m++)
        for(int i=1;i<=E->lmesh.nox;i++)
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int k=1;k<=E->lmesh.noz;k++)
                {
                    int node = k + (i-1)*E->lmesh.noz
			     + (j-1)*E->lmesh.nox*E->lmesh.noz;

		    for(int d=0; d<DIM; d++)
			x[d] = E->sx[m][d+1][node];

                    if(isInside(x, bbox_)) {
                        X_.push_back(x);
                        nodeID_.push_back(node);
                    }
                }
}


// version
// $Id: Interior.cc,v 1.9 2003/12/30 21:46:01 tan2 Exp $

// End of file
