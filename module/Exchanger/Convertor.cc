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
#include "journal/diagnostics.h"


void Convertor::init(bool dimensional, bool transformational,
		     const All_variables* E)
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    if(dimensional)
	si = new SIUnit(E);

    if(transformational)
	cart = new Exchanger::Spherical2Cartesian();

    inited = true;
}


// version
// $Id: Convertor.cc,v 1.6 2005/06/03 21:51:42 leif Exp $

// End of file
