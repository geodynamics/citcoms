// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
//#include "Sink.h"
#include "Inlet.h"


Inlet::Inlet(const BoundedMesh& boundedMesh,
	     const Sink& s,
	     All_variables* e) :
    mesh(boundedMesh),
    sink(s),
    E(e)
{}


Inlet::~Inlet()
{}



// version
// $Id: Inlet.cc,v 1.1 2004/02/24 20:03:09 tan2 Exp $

// End of file
