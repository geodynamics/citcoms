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
#include "SIUnit.h"


SIUnit::SIUnit(const All_variables* E) :
    Exchanger::SIUnit() 
{
    length_factor = E->data.radius_km * 1000.;
    velocity_factor = E->data.therm_diff / length_factor;
    temperature_factor = E->data.ref_temperature;
    temperature_offset = E->data.surf_temp;
    time_factor = length_factor / velocity_factor;
    traction_factor = E->data.ref_viscosity * E->data.therm_diff;
}


SIUnit::~SIUnit()
{}


// version
// $Id: SIUnit.cc,v 1.5 2004/05/11 07:55:30 tan2 Exp $

// End of file
