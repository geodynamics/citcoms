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
			       All_variables* E) :
    Outlet(source, E),
    f(source.size())
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << " source.size = " << source.size()
	  << journal::end;
}


TractionOutlet::~TractionOutlet()
{}


void TractionOutlet::send()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    source.interpolateTraction(f);

    Convertor& convertor = Convertor::instance();
    convertor.traction(f, source.getX());

    f.print( "TractionOutlet_F" );
    source.send(f);
}


// version
// $Id: TractionOutlet.cc,v 1.1 2004/02/24 20:14:21 tan2 Exp $

// End of file
