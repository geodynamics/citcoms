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
    Boundary(bool dimensional, bool transformational);
    explicit Boundary(const All_variables* E, bool dimensional, bool transformational);
    virtual ~Boundary() {};

private:
    void initBBox(const All_variables *E);
    void initX(const All_variables *E);
    inline bool isOnBoundary(const All_variables* E, int i,
			     int j, int k) const;

};


#endif

// version
// $Id: Boundary.h,v 1.26 2004/01/06 22:40:28 puru Exp $

// End of file
