// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "journal/diagnostics.h"
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
    debug << journal::at(__HERE__) << journal::endl;
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
// $Id: SVTOutlet.cc,v 1.4 2005/06/03 21:51:42 leif Exp $

// End of file
