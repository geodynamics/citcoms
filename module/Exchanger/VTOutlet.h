// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_VTOutlet_h)
#define pyCitcom_VTOutlet_h

#include <string>
#include "Array2D.h"
#include "DIM.h"
#include "Outlet.h"

struct All_variables;
class BoundedMesh;
class VTSource;


class VTOutlet : public Outlet {
    const bool modeV;
    const bool modeT;
    Array2D<double,DIM> v;
    Array2D<double,1> t;

public:
    VTOutlet(const VTSource& source,
	     All_variables* E,
	     const std::string& mode);
    virtual ~VTOutlet();

    virtual void send();

private:
    void sendVT();
    void sendV();
    void sendT();

    // disable copy c'tor and assignment operator
    VTOutlet(const VTOutlet&);
    VTOutlet& operator=(const VTOutlet&);

};


#endif

// version
// $Id: VTOutlet.h,v 1.1 2004/02/24 20:24:21 tan2 Exp $

// End of file
