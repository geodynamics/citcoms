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
#include "journal/diagnostics.h"
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
    debug << journal::at(__HERE__) << journal::endl;
}


BoundaryVTInlet::~BoundaryVTInlet()
{
    delete awnormal;
}


void BoundaryVTInlet::recv()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    VTInlet::recv();

    awnormal->imposeConstraint(v, comm, sink, comm);
    v.print("CitcomS-BoundaryVTInlet-V_constrained");
}


// version
// $Id: BoundaryVTInlet.cc 15108 2009-06-02 22:56:46Z tan2 $

// End of file
