// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "Exchanger/Spherical2Cartesian.h"
#include "SIUnit.h"
#include "Convertor.h"


void Convertor::init(bool dimensional, bool transformational,
		     const All_variables* E)
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    if(dimensional)
	si = new SIUnit(E);

    if(transformational)
	cart = new Exchanger::Spherical2Cartesian();

    inited = true;
}


// version
// $Id: Convertor.cc,v 1.5 2004/05/11 07:55:30 tan2 Exp $

// End of file
