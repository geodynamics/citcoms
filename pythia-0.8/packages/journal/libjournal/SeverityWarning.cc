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
#include "SeverityWarning.h"

#include "Index.h"
#include "FacilityMap.h"

using namespace journal;

// interface
SeverityWarning::state_t & SeverityWarning::lookup(string_t name) {
    return _index->lookup(name);
}

// meta-methods
SeverityWarning::~SeverityWarning() {}

// data
SeverityWarning::index_t * SeverityWarning::_index = new FacilityMap(true);

// version
// $Id: SeverityWarning.cc,v 1.1.1.1 2005/03/08 16:13:56 aivazis Exp $

// End of file 
