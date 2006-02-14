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
#include "Index.h"
#include "FacilityMap.h"
#include "Facility.h"

using namespace journal;

// interface
FacilityMap::state_t & FacilityMap::lookup(string_t name) {
    state_t * state;
    index_t::const_iterator target = _index.find(name);

    if (target == _index.end()) {
        state = new Facility(_defaultState);
        _index.insert(entry_t(name, state));
    } else {
        state = (*target).second;
    }

    return *state;
}

// meta-methods
FacilityMap::~FacilityMap() {}

// version
// $Id: FacilityMap.cc,v 1.1.1.1 2005/03/08 16:13:56 aivazis Exp $

// End of file 
