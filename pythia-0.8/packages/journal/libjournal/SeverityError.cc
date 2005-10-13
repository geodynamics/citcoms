// -*- C++ -*-
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                             Michael A.G. Aivazis
//                      California Institute of Technology
//                      (C) 1998-2005  All Rights Reserved
//
// <LicenseText>
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>

#include <map>
#include <string>
#include <sstream>

#include "Diagnostic.h"
#include "SeverityError.h"

#include "Index.h"
#include "FacilityMap.h"

using namespace journal;

// interface
SeverityError::state_t & SeverityError::lookup(string_t name) {
    return _index->lookup(name);
}

// meta-methods
SeverityError::~SeverityError() {}

// data
SeverityError::index_t * SeverityError::_index = new FacilityMap(true);

// version
// $Id: SeverityError.cc,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
