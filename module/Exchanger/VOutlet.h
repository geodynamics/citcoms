// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_VOutlet_h)
#define pyCitcomSExchanger_VOutlet_h

#include "Exchanger/Outlet.h"

struct All_variables;
class CitcomSource;


class VOutlet : public Exchanger::Outlet {
    All_variables* E;
    Exchanger::Array2D<double,Exchanger::DIM> v;

public:
    VOutlet(const CitcomSource& source,
	     All_variables* E);
    virtual ~VOutlet();

    virtual void send();

private:
    // disable copy c'tor and assignment operator
    VOutlet(const VOutlet&);
    VOutlet& operator=(const VOutlet&);

};


#endif

// version
// $Id: VOutlet.h,v 1.1 2004/05/18 21:18:13 ces74 Exp $

// End of file
