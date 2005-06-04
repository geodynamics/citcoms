// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "global_defs.h"
#include "journal/diagnostics.h"
#include "CitcomSource.h"
#include "Convertor.h"
#include "VOutlet.h"


VOutlet::VOutlet(const CitcomSource& source,
		   All_variables* e) :
    Outlet(source),
    E(e),
    v(source.size())
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;
}


VOutlet::~VOutlet()
{}


void VOutlet::send()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    source.interpolateVelocity(v);
    v.print("CitcomS-VOutlet-V");

    Exchanger::Convertor& convertor = Convertor::instance();
    convertor.velocity(v, source.getX());

    source.send(v);
}


// private functions



// version
// $Id: VOutlet.cc,v 1.2 2005/06/03 21:51:42 leif Exp $

// End of file
