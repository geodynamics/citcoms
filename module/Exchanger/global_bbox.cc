// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

#include <portinfo>
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
// $Id: global_bbox.cc,v 1.2 2004/07/02 20:47:01 tan2 Exp $

// End of file
