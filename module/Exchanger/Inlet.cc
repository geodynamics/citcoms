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
    E(e),
    fge_t(0),
    cge_t(0)
{}


Inlet::~Inlet()
{}


void Inlet::storeTimestep(double fge_time, double cge_time)
{
    fge_t = fge_time;
    cge_t = cge_time;
}


// protected functions

void Inlet::getFactor(double& N1, double& N2) const
{
    if(cge_t == 0) {
        N1 = 0.0;
        N2 = 1.0;
    } else {
        N1 = (cge_t - fge_t) / cge_t;
        N2 = fge_t / cge_t;
    }
}


// version
// $Id: Inlet.cc,v 1.2 2004/03/11 01:06:14 tan2 Exp $

// End of file
