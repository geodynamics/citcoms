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
#include "Convertor.h"
#include "VTSource.h"
#include "VTOutlet.h"


VTOutlet::VTOutlet(const VTSource& source,
		   All_variables* E,
		   const std::string& mode) :
    Outlet(source, E),
    modeV(mode.find('V',0) != std::string::npos),
    modeT(mode.find('T',0) != std::string::npos)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "modeV = " << modeV << "  modeT = " << modeT
	  << journal::end;

    if(!(modeV || modeT)) {
	journal::firewall_t firewall("VTOutlet");
	firewall << journal::loc(__HERE__)
		 << "invalid mode" << journal::end;
    }

    if(modeV)
	v.resize(source.size());

    if(modeT)
	t.resize(source.size());
}


VTOutlet::~VTOutlet()
{}


void VTOutlet::send()
{
    if(modeV && modeT)
	sendVT();
    else if(modeV)
	sendV();
    else
	sendT();
}


// private functions

void VTOutlet::sendVT()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    source.interpolateVelocity(v);
    v.print("VTOutlet_V");

    source.interpolateTemperature(t);
    t.print("VTOutlet_T");

    Convertor& convertor = Convertor::instance();
    convertor.velocity(v, source.getX());
    convertor.temperature(t);

    source.send(t, v);
}


void VTOutlet::sendV()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    source.interpolateVelocity(v);
    v.print("VTOutlet_V");

    Convertor& convertor = Convertor::instance();
    convertor.velocity(v, source.getX());

    source.send(v);
}


void VTOutlet::sendT()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    source.interpolateTemperature(t);
    t.print("VTOutlet_T");

    Convertor& convertor = Convertor::instance();
    convertor.temperature(t);

    source.send(t);
}


// private functions



// version
// $Id: VTOutlet.cc,v 1.1 2004/02/24 20:24:21 tan2 Exp $

// End of file
