// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_Transformational_h)
#define pyCitcom_Transformational_h

#include "Array2D.h"
#include "BoundedBox.h"
#include "DIM.h"

struct All_variables;

class Transformational {
    
     static const All_variables* E;
    
public:
    ~Transformational() {};

    static void setE(const All_variables* E);
    static Transformational& instance();
    
    // Transform to cartesian coordinate system from spherical system
    void coordinate(BoundedBox& bbox) const;
    void coordinate(Array2D<double,DIM>& X) const;
    void traction(Array2D<double,DIM>& F) const;
    void velocity(Array2D<double,DIM>& V) const;

    // Transform to spherical coordinate system from cartesian system
    void xcoordinate(BoundedBox& bbox) const;
    void xcoordinate(Array2D<double,DIM>& X) const;
    void xtraction(Array2D<double,DIM>& F) const;
    void xvelocity(Array2D<double,DIM>& V) const;

private:
    Transformational();
};




#endif

// version
// $Id: Transformational.h,v 1.1 2004/01/06 22:38:07 puru Exp $

// End of file
