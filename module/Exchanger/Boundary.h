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
// $Id: Boundary.h,v 1.24 2003/11/11 19:29:27 tan2 Exp $

// End of file
