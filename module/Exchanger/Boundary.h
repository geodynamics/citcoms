// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_Boundary_h)
#define pyCitcom_Boundary_h

#include "BoundedMesh.h"

struct All_variables;


class Boundary : public BoundedMesh {

public:
    Boundary();
    explicit Boundary(const All_variables* E);
    virtual ~Boundary() {};

private:
    void initBBox(const All_variables *E);
    void initX(const All_variables *E);
    inline bool isOnBoundary(const All_variables* E, int i,
			     int j, int k) const;

};


#endif

// version
// $Id: Boundary.h,v 1.27 2004/01/07 21:54:00 tan2 Exp $

// End of file
