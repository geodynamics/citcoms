// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_TractionBC_h)
#define pyCitcom_TractionBC_h

#include "AreaWeightedNormal.h"
#include "Array2D.h"
#include "DIM.h"

struct All_variables;
class Boundary;
class TractionSource;


class TractionBC {
    All_variables* E;
    TractionSource& source;
    Array2D<double,DIM> fbc;

public:
    TractionBC(TractionSource& src, All_variables* E);
    ~TractionBC() {};

    void sendTraction();
    void domain_cutout();

private:

};


#endif

// version
// $Id: TractionBC.h,v 1.1 2004/01/14 02:11:24 ces74 Exp $

// End of file
