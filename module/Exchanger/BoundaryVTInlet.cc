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

#include <portinfo>
#include "journal/journal.h"
#include "AreaWeightedNormal.h"
#include "Boundary.h"
#include "BoundaryVTInlet.h"

using Exchanger::Sink;


BoundaryVTInlet::BoundaryVTInlet(const Boundary& boundary,
				 const Sink& sink,
				 All_variables* E,
				 MPI_Comm c) :
    VTInlet(boundary, sink, E),
    comm(c),
    awnormal(new AreaWeightedNormal(comm, boundary, sink, E))
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;
}


BoundaryVTInlet::~BoundaryVTInlet()
{
    delete awnormal;
}


void BoundaryVTInlet::recv()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    VTInlet::recv();

    awnormal->imposeConstraint(v, comm, sink);
    v.print("CitcomS-BoundaryVTInlet-V_constrained");
}


// version
// $Id: BoundaryVTInlet.cc 2397 2005-10-04 22:37:25Z leif $

// End of file
