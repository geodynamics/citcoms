// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_CartesianCoord_h)
#define pyCitcom_CartesianCoord_h

#include "Array2D.h"
#include "BoundedBox.h"
#include "DIM.h"

class BoundedMesh;


class CartesianCoord {

public:
    CartesianCoord() {};
    ~CartesianCoord() {};

    // Transform to cartesian coordinate system from spherical system
    void coordinate(BoundedBox& bbox) const;
    void coordinate(Array2D<double,DIM>& X) const;
    void vector(Array2D<double,DIM>& V,
		const Array2D<double,DIM>& X) const;

    // Transform to spherical coordinate system from cartesian system
    void xcoordinate(BoundedBox& bbox) const;
    void xcoordinate(Array2D<double,DIM>& X) const;
    void xvector(Array2D<double,DIM>& V,
		 const Array2D<double,DIM>& X) const;

private:

};


#endif

// version
// $Id: CartesianCoord.h,v 1.2 2004/01/07 21:54:00 tan2 Exp $

// End of file
