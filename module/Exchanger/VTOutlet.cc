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
#include "VTOutlet.h"


VTOutlet::VTOutlet(const CitcomSource& source,
		   All_variables* e) :
    Outlet(source),
    E(e),
    v(source.size()),
    t(source.size())

{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;
}


VTOutlet::~VTOutlet()
{}


void VTOutlet::send()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    source.interpolateVelocity(v);
    v.print("CitcomS-VTOutlet-V");

    source.interpolateTemperature(t);
    t.print("CitcomS-VTOutlet-T");

    Exchanger::Convertor& convertor = Convertor::instance();
    convertor.velocity(v, source.getX());
    convertor.temperature(t);

    source.send(t, v);
}


// private functions



// version
// $Id: VTOutlet.cc,v 1.3 2005/06/03 21:51:42 leif Exp $

// End of file
