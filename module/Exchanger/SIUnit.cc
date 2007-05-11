// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//
// CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
// Copyright (C) 2002-2005, California Institute of Technology.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//</LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include "config.h"
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
    stress_factor = E->data.ref_viscosity * E->data.therm_diff / (length_factor*length_factor);
}


SIUnit::~SIUnit()
{}


// version
// $Id$

// End of file
