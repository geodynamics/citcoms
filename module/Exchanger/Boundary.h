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
    Boundary(bool dimensional);
    explicit Boundary(const All_variables* E, bool dimensional);
    virtual ~Boundary() {};

private:
    void initBBox(const All_variables *E);
    void initX(const All_variables *E);
    inline bool isOnBoundary(const All_variables* E, int i,
			     int j, int k) const;

};


#endif

// version
// $Id: Boundary.h,v 1.25 2003/12/30 21:46:01 tan2 Exp $

// End of file
