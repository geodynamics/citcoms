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
    Array2D<int,DIM> normal_;

public:
    Boundary();
    explicit Boundary(const All_variables* E);
    virtual ~Boundary() {};

    inline int normal(int d, int n) const {return normal_[d][n];}

private:
    void initBBox(const All_variables *E);
    void initX(const All_variables *E);

};


#endif

// version
// $Id: Boundary.h,v 1.28 2004/03/11 22:50:09 tan2 Exp $

// End of file
