// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_FEMInterpolator_h)
#define pyCitcom_FEMInterpolator_h

#include "Array2D.h"
#include "DIM.h"


class FEMInterpolator {
protected:
    Array2D<int,1> elem_;  // elem # from which fields are interpolated
    Array2D<double,NODES_PER_ELEMENT> shape_; // shape functions for interpolation

public:
    FEMInterpolator(int n) : elem_(n), shape_(n) {};
    virtual ~FEMInterpolator() {};

    inline int size() const {return elem_.size();}
    virtual void interpolateDisplacement(Array2D<double,DIM>& D) const = 0;
    virtual void interpolateForce(Array2D<double,DIM>& F) const = 0;
    virtual void interpolatePressure(Array2D<double,1>& P) const = 0;
    virtual void interpolateStress(Array2D<double,DIM>& S) const = 0;
    virtual void interpolateTemperature(Array2D<double,1>& T) const = 0;
    virtual void interpolateTraction(Array2D<double,DIM>& F) const = 0;
    virtual void interpolateVelocity(Array2D<double,DIM>& V) const = 0;

private:
    // disable copy c'tor and assignment operator
    FEMInterpolator(const FEMInterpolator&);
    FEMInterpolator& operator=(const FEMInterpolator&);

};



#endif

// version
// $Id: FEMInterpolator.h,v 1.1 2003/11/25 02:59:11 tan2 Exp $

// End of file
