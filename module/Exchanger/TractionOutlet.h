// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_TractionOutlet_h)
#define pyCitcom_TractionOutlet_h

#include "Array2D.h"
#include "DIM.h"
#include "Outlet.h"

struct All_variables;
class TractionSource;


class TractionOutlet : public Outlet {
    Array2D<double,DIM> f;

public:
    TractionOutlet(TractionSource& src, All_variables* E);
    virtual ~TractionOutlet();

    virtual void send();

private:

};


#endif

// version
// $Id: TractionOutlet.h,v 1.1 2004/02/24 20:14:21 tan2 Exp $

// End of file
