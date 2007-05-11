// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//
// CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
// Copyright (C) 2002-2005, California Institute of Technology.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//</LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include "config.h"
#include <algorithm>
#include <limits>
#include <vector>
#include "global_defs.h"
#include "Interior.h"
#include "journal/diagnostics.h"


Interior::Interior() :
    Exchanger::BoundedMesh()
{}



Interior::Interior(const Exchanger::BoundedBox& remoteBBox,
		   const All_variables* E) :
    Exchanger::BoundedMesh()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

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
// $Id$

// End of file
