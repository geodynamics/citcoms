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
#include "journal/journal.h"
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
    debug << journal::loc(__HERE__) << journal::end;
}


VOutlet::~VOutlet()
{}


void VOutlet::send()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    source.interpolateVelocity(v);
    v.print("CitcomS-VOutlet-V");

    Exchanger::Convertor& convertor = Convertor::instance();
    convertor.velocity(v, source.getX());

    source.send(v);
}


// private functions



// version
// $Id: VOutlet.cc,v 1.1 2004/05/18 21:18:13 ces74 Exp $

// End of file
