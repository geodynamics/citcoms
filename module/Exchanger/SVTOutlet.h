// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
// Role:
//     send stress, velocity and temperature to sink
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_SVTOutlet_h)
#define pyCitcomSExchanger_SVTOutlet_h

#include "Exchanger/Outlet.h"

struct All_variables;
class CitcomSource;


class SVTOutlet : public Exchanger::Outlet {
    All_variables* E;
    Exchanger::Array2D<double,Exchanger::STRESS_DIM> s;
    Exchanger::Array2D<double,Exchanger::DIM> v;
    Exchanger::Array2D<double,1> t;

public:
    SVTOutlet(const CitcomSource& source,
	      All_variables* E);
    virtual ~SVTOutlet();

    virtual void send();

private:
    // disable copy c'tor and assignment operator
    SVTOutlet(const SVTOutlet&);
    SVTOutlet& operator=(const SVTOutlet&);

};


#endif

// version
// $Id: SVTOutlet.h,v 1.2 2004/05/11 07:55:30 tan2 Exp $

// End of file
