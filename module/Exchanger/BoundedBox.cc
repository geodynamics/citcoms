// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <vector>
#include "global_defs.h"
#include "BoundedBox.h"
#include "dimensionalization.h"

bool isOverlapped(const BoundedBox& lhs,
		  const BoundedBox& rhs)
{
    std::vector<double> x(DIM);

    // loop over all vertices of rhs, assuming DIM==3
    for(int i=0; i<2; i++)
	for(int j=0; j<2; j++)
	    for(int k=0; k<2; k++) {

		x[0] = lhs[i][0];
		x[1] = lhs[j][1];
		x[2] = lhs[k][2];

		if(isInside(x, rhs)) return true;
	    }
    return false;
}


bool isInside(const std::vector<double>& x,
	      const BoundedBox& bbox)
{
    bool inside = true;
    for(int m=0; m<DIM; m++)
	if(x[m] < bbox[0][m] ||
	   x[m] > bbox[1][m]) {
	    inside = false;
	    break;
	}

    return inside;
}


void fullGlobalBoundedBox(BoundedBox& bbox, const All_variables* E)
{
    const double pi = std::atan(1.0) * 4;
    bbox[0][0] = bbox[0][1] = 0;
    bbox[1][0] = 2 * pi;
    bbox[1][1] = pi;

    bbox[0][2] = E->sphere.ri*dimensional_len;
    bbox[1][2] = E->sphere.ro*dimensional_len;
}


void regionalGlobalBoundedBox(BoundedBox& bbox, const All_variables* E)
{
    bbox[0][0] = E->control.theta_min;
    bbox[1][0] = E->control.theta_max;
    bbox[0][1] = E->control.fi_min;
    bbox[1][1] = E->control.fi_max;
    bbox[0][2] = E->sphere.ri*dimensional_len;
    bbox[1][2] = E->sphere.ro*dimensional_len;
}


// version
// $Id: BoundedBox.cc,v 1.3 2003/12/17 04:27:56 puru Exp $

// End of file
