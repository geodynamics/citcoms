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
#include "Boundary.h"
#include "Convertor.h"
#include "TractionSource.h"
#include "TractionBC.h"

extern "C" {
    void check_bc_consistency(const All_variables *E);
    void construct_id(const All_variables *E);
    void temperatures_conform_bcs(All_variables* E);
}



//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


TractionBC::TractionBC(TractionSource& s,
						 All_variables* e) :
    E(e),
    source(s),
    fbc(source.size())
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in TractionBC::Citcom"
	  << " source.size = " << source.size()
	  << journal::end;
}


void TractionBC::sendTraction()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in TractionBC::sendTraction" << journal::end;

    source.interpolateTraction(fbc);

    debug << journal::loc(__HERE__)
	  << "after interpolateTration" << journal::end;


    Convertor& convertor = Convertor::instance();
    convertor.traction(fbc, source.getX());

    fbc.print( "FBC" );
    source.send(fbc);
}

void TractionBC::domain_cutout()
{

    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in TractionBC::domain_cutout" << journal::end;

	source.domain_cutout();
}
// version
// $Id: TractionBC.cc,v 1.3 2004/01/15 02:59:16 ces74 Exp $

// End of file
