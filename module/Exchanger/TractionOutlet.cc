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
#include "Convertor.h"
#include "TractionSource.h"
#include "TractionOutlet.h"

extern "C" {
    void check_bc_consistency(const All_variables *E);
    void construct_id(const All_variables *E);
    void temperatures_conform_bcs(All_variables* E);
}



//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


TractionOutlet::TractionOutlet(TractionSource& source,
			       All_variables* E,
			       const std::string& mode) :
    Outlet(source, E),
    modeF(mode.find('F',0) != std::string::npos),
    modeV(mode.find('V',0) != std::string::npos)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "modeF = " << modeF << "  modeV = " << modeV
	  << journal::end;

    if(modeF)
	f.resize(source.size());

    if(modeV)
	v.resize(source.size());
}


TractionOutlet::~TractionOutlet()
{}


void TractionOutlet::send()
{
    if(modeF && modeV)
        sendFV();
    else if(modeV)
        sendV();
    else
        sendF();
}


// private functions

void TractionOutlet::sendFV()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    source.interpolateTraction(f);
    source.interpolateVelocity(v);

    Convertor& convertor = Convertor::instance();
    convertor.traction(f, source.getX());
    convertor.velocity(v, source.getX());

    f.print( "TractionOutlet_F" );
    v.print( "TractionOutlet_V" );
    source.send(f, v);
}


void TractionOutlet::sendF()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    source.interpolateTraction(f);

    Convertor& convertor = Convertor::instance();
    convertor.traction(f, source.getX());

    f.print( "TractionOutlet_F" );
    source.send(f);
}


void TractionOutlet::sendV()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    source.interpolateVelocity(v);

    Convertor& convertor = Convertor::instance();
    convertor.velocity(v, source.getX());

    v.print( "TractionOutlet_V" );
    source.send(v);
}


// version
// $Id: TractionOutlet.cc,v 1.2 2004/03/28 23:19:00 tan2 Exp $

// End of file
