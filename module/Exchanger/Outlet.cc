// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "AbstractSource.h"
#include "Outlet.h"


Outlet::Outlet(const AbstractSource& src, All_variables* e) :
    source(src),
    E(e)
{}


Outlet::~Outlet()
{}


// version
// $Id: Outlet.cc,v 1.1 2004/02/24 20:03:09 tan2 Exp $

// End of file
