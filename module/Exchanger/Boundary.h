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
    void initX(const All_variables *E);
    void appendX(const All_variables *E, int m, int node);

};


#endif

// version
// $Id: Boundary.h,v 1.22 2003/11/07 01:08:01 tan2 Exp $

// End of file
