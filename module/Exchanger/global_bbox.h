// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_global_bbox_h)
#define pyCitcomSExchanger_global_bbox_h

#include "Exchanger/BoundedBox.h"

struct All_variables;


void fullGlobalBoundedBox(Exchanger::BoundedBox& bbox,
			  const All_variables* E);
void regionalGlobalBoundedBox(Exchanger::BoundedBox& bbox,
			      const All_variables* E);


#endif

// version
// $Id: global_bbox.h,v 1.1 2004/05/11 07:55:30 tan2 Exp $

// End of file
