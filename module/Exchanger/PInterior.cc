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
#include <vector>
#include "global_defs.h"
#include "PInterior.h"
#include "journal/diagnostics.h"


PInterior::PInterior() :
    Exchanger::BoundedMesh()
{}



PInterior::PInterior(const Exchanger::BoundedBox& bbox,
                     const All_variables* E) :
    Exchanger::BoundedMesh()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    bbox_ = bbox;
    bbox_.print("CitcomS-PInterior-BBox");

    X_.reserve(E->lmesh.nel);
    nodeID_.reserve(E->lmesh.nel);

    initX(E);

    X_.shrink();
    X_.print("CitcomS-PInterior-X");

    nodeID_.shrink();
    nodeID_.print("CitcomS-PInterior-nodeID");
}


void PInterior::initX(const All_variables* E)
{
    std::vector<double> x(Exchanger::DIM);

    // Storing the coordinates of the center of elements 
    for (int m=1;m<=E->sphere.caps_per_proc;m++)
        for(int e=1;e<=E->lmesh.nel;e++) {
            for(int d=0; d<Exchanger::DIM; d++)
                x[d] = E->eco[m][e].centre[d+1];
            
            if(isInside(x, bbox_)) {
                X_.push_back(x);
                nodeID_.push_back(e);
            }
        }
}


// version
// $Id$

// End of file
