// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_initTemperature_h)
#define pyCitcom_initTemperature_h

#include "BoundedBox.h"

struct All_variables;
class Interior;
class Sink;
class Source;

void initTemperatureSink(const Interior& interior,
			 const Sink& sink,
			 All_variables* E);
void initTemperatureSource(const Source& source,
			   All_variables* E);
void modifyT(const BoundedBox& bbox, All_variables* E);


#endif

// version
// $Id: initTemperature.h,v 1.2 2003/12/19 08:05:13 tan2 Exp $

// End of file
