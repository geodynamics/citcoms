// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_VTInterpolator_h)
#define pyCitcom_VTInterpolator_h

#include "FEMInterpolator.h"

struct All_variables;
class BoundedMesh;


class VTInterpolator: public FEMInterpolator {

public:
    VTInterpolator(const BoundedMesh& boundedMesh,
		   const All_variables* E,
		   Array2D<int,1>& meshNode);
    virtual ~VTInterpolator() {};

    virtual void interpolateTemperature(Array2D<double,1>& T);
    virtual void interpolateVelocity(Array2D<double,DIM>& V);

private:
    // disable these functions
    virtual void interpolateDisplacement(Array2D<double,DIM>& D) {};
    virtual void interpolateForce(Array2D<double,DIM>& F) {};
    virtual void interpolatePressure(Array2D<double,1>& P) {};
    virtual void interpolateStress(Array2D<double,DIM>& S) {};
    virtual void interpolateTraction(Array2D<double,DIM>& F) {};

};



#endif

// version
// $Id: VTInterpolator.h,v 1.1 2004/01/08 20:42:56 tan2 Exp $

// End of file
