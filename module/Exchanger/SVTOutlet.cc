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
#include "Convertor.h"
#include "VTSource.h"
#include "SVTOutlet.h"


SVTOutlet::SVTOutlet(const VTSource& source,
		     All_variables* E) :
    Outlet(source, E)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    s.resize(source.size());
    v.resize(source.size());
    t.resize(source.size());
}


SVTOutlet::~SVTOutlet()
{}


void SVTOutlet::send()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    source.interpolateStress(s);
    s.print("SVTOutlet_S");

    source.interpolateVelocity(v);
    v.print("SVTOutlet_V");

    source.interpolateTemperature(t);
    t.print("SVTOutlet_T");

    Convertor& convertor = Convertor::instance();
    //convertor.stress(s, source.getX());
    convertor.velocity(v, source.getX());
    convertor.temperature(t);

    source.send(t, v);
    source.send(s);
}


// version
// $Id: SVTOutlet.cc,v 1.1 2004/04/16 00:03:50 tan2 Exp $

// End of file
