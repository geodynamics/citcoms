// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_TOutlet_h)
#define pyCitcomSExchanger_TOutlet_h

#include "Exchanger/Outlet.h"

struct All_variables;
class CitcomSource;


class TOutlet : public Exchanger::Outlet {
    All_variables* E;
    Exchanger::Array2D<double,1> t;

public:
    TOutlet(const CitcomSource& source,
	    All_variables* E);
    virtual ~TOutlet();

    virtual void send();

private:
    // disable copy c'tor and assignment operator
    TOutlet(const TOutlet&);
    TOutlet& operator=(const TOutlet&);

};


#endif

// version
// $Id: TOutlet.h,v 1.1 2004/05/11 07:55:30 tan2 Exp $

// End of file
