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
#include "TOutlet.h"


TOutlet::TOutlet(const CitcomSource& source,
		 All_variables* e) :
    Outlet(source),
    E(e),
    t(source.size())
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;
}


TOutlet::~TOutlet()
{}


void TOutlet::send()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    source.interpolateTemperature(t);
    t.print("CitcomS-TOutlet-T");

    Exchanger::Convertor& convertor = Convertor::instance();
    convertor.temperature(t);

    source.send(t);
}


// private functions



// version
// $Id: TOutlet.cc,v 1.1 2004/05/11 07:55:30 tan2 Exp $

// End of file
