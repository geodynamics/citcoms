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

#if !defined(pyCitcom_SVTOutlet_h)
#define pyCitcom_SVTOutlet_h

#include <string>
#include "Array2D.h"
#include "DIM.h"
#include "Outlet.h"

struct All_variables;
class BoundedMesh;
class VTSource;


class SVTOutlet : public Outlet {
    Array2D<double,STRESS_DIM> s;
    Array2D<double,DIM> v;
    Array2D<double,1> t;

public:
    SVTOutlet(const VTSource& source,
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
// $Id: SVTOutlet.h,v 1.1 2004/04/16 00:03:50 tan2 Exp $

// End of file
