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
#include "SeverityDebug.h"

#include "Index.h"
#include "FacilityMap.h"

using namespace journal;

// interface
SeverityDebug::state_t & SeverityDebug::lookup(string_t name) {
    return _index->lookup(name);
}

// meta-methods
SeverityDebug::~SeverityDebug() {}

// data
SeverityDebug::index_t * SeverityDebug::_index = new FacilityMap(false);

// version
// $Id: SeverityDebug.cc,v 1.1.1.1 2005/03/08 16:13:56 aivazis Exp $

// End of file 
