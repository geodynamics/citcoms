// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_BoundedBox_h)
#define pyCitcom_BoundedBox_h

#include "Array2D.h"
#include "DIM.h"

struct All_variables;

typedef Array2D<double,2> BoundedBox;

bool isOverlapped(const BoundedBox& lhs,
		  const BoundedBox& rhs);

bool isInside(const std::vector<double>& x,
	      const BoundedBox& bbox);

void fullGlobalBoundedBox(BoundedBox& bbox, const All_variables* E);
void regionalGlobalBoundedBox(BoundedBox& bbox, const All_variables* E);

#endif

// version
// $Id: BoundedBox.h,v 1.1 2003/11/07 01:08:01 tan2 Exp $

// End of file
