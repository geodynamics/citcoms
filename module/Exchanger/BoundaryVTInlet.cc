// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
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
// $Id: BoundaryVTInlet.cc,v 1.2 2004/05/11 18:35:24 tan2 Exp $

// End of file
