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
    virtual void interpolateDisplacement(Array2D<double,DIM>& D) = 0;
    virtual void interpolateForce(Array2D<double,DIM>& F) = 0;
    virtual void interpolatePressure(Array2D<double,1>& P) = 0;
    virtual void interpolateStress(Array2D<double,DIM>& S) = 0;
    virtual void interpolateTemperature(Array2D<double,1>& T) = 0;
    virtual void interpolateTraction(Array2D<double,DIM>& F) = 0;
    virtual void interpolateVelocity(Array2D<double,DIM>& V) = 0;

private:
    // disable copy c'tor and assignment operator
    FEMInterpolator(const FEMInterpolator&);
    FEMInterpolator& operator=(const FEMInterpolator&);

};



#endif

// version
// $Id: FEMInterpolator.h,v 1.2 2003/12/16 02:14:10 tan2 Exp $

// End of file
