// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
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
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

#include <portinfo>
#include <cmath>
#include "global_defs.h"
#include "global_bbox.h"


void fullGlobalBoundedBox(Exchanger::BoundedBox& bbox,
			  const All_variables* E)
{
    const double pi = std::atan(1.0) * 4;
    bbox[0][0] = bbox[0][1] = 0;
    bbox[1][0] = pi;
    bbox[1][1] = 2 * pi;

    bbox[0][2] = E->sphere.ri;
    bbox[1][2] = E->sphere.ro;
}


void regionalGlobalBoundedBox(Exchanger::BoundedBox& bbox,
			      const All_variables* E)
{
    bbox[0][0] = E->control.theta_min;
    bbox[1][0] = E->control.theta_max;
    bbox[0][1] = E->control.fi_min;
    bbox[1][1] = E->control.fi_max;
    bbox[0][2] = E->sphere.ri;
    bbox[1][2] = E->sphere.ro;

}


// version
// $Id: global_bbox.cc 2397 2005-10-04 22:37:25Z leif $

// End of file
