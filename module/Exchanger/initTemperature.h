// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_initTemperature_h)
#define pyCitcomSExchanger_initTemperature_h

void initTemperature(const Exchanger::BoundedBox&, All_variables* E);
void modifyT(const Exchanger::BoundedBox& bbox, All_variables* E);

#endif

// version
// $Id: initTemperature.h,v 1.4 2004/05/11 07:55:30 tan2 Exp $

// End of file
