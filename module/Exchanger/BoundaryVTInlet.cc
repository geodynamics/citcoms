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


BoundaryVTInlet::BoundaryVTInlet(MPI_Comm c,
				 const Boundary& boundary,
				 const Sink& sink,
				 All_variables* E,
				 const std::string& mode) :
    VTInlet(boundary, sink, E, mode),
    comm(c),
    awnormal(new AreaWeightedNormal(comm, boundary, sink, E))
{
    if(mode.find('V',0) == std::string::npos) {
	journal::firewall_t firewall("BoundaryVTInlet");
	firewall << journal::loc(__HERE__)
		 << "invalid mode" << journal::end;
    }
}


BoundaryVTInlet::~BoundaryVTInlet()
{
    delete awnormal;
}


void BoundaryVTInlet::recv()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    VTInlet::recv();

    awnormal->imposeConstraint(v, comm, sink);
    v.print("V_constrained");
}


// version
// $Id: BoundaryVTInlet.cc,v 1.1 2004/02/24 20:34:43 tan2 Exp $

// End of file
