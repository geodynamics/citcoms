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
#include "CitcomSource.h"
#include "SVTOutlet.h"

using Exchanger::Array2D;
using Exchanger::DIM;
using Exchanger::STRESS_DIM;


SVTOutlet::SVTOutlet(const CitcomSource& source,
		     All_variables* e) :
    Outlet(source),
    E(e),
    s(source.size()),
    v(source.size()),
    t(source.size())
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;
}


SVTOutlet::~SVTOutlet()
{}


void SVTOutlet::send()
{
//     journal::debug_t debug("CitcomS-Exchanger");
//     debug << journal::loc(__HERE__) << journal::end;

    source.interpolateStress(s);
//     s.print("CitcomS-SVTOutlet-S");

    source.interpolateVelocity(v);
//     v.print("CitcomS-SVTOutlet-V");

    source.interpolateTemperature(t);
//     t.print("CitcomS-SVTOutlet-T");

    Exchanger::Convertor& convertor = Convertor::instance();
    //convertor.stress(s, source.getX());
    convertor.velocity(v, source.getX());
    convertor.temperature(t);

    source.send(t, v);
    source.send(s);
}


// version
// $Id: SVTOutlet.cc,v 1.3 2004/05/25 00:29:30 tan2 Exp $

// End of file
