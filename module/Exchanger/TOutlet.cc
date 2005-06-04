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
#include "TOutlet.h"


TOutlet::TOutlet(const CitcomSource& source,
		 All_variables* e) :
    Outlet(source),
    E(e),
    t(source.size())
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;
}


TOutlet::~TOutlet()
{}


void TOutlet::send()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    source.interpolateTemperature(t);
    t.print("CitcomS-TOutlet-T");

    Exchanger::Convertor& convertor = Convertor::instance();
    convertor.temperature(t);

    source.send(t);
}


// private functions



// version
// $Id: TOutlet.cc,v 1.2 2005/06/03 21:51:42 leif Exp $

// End of file
