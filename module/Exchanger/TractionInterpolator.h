// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_TractionInterpolator_h)
#define pyCitcom_TractionInterpolator_h

#include "Array2D.h"
#include "DIM.h"
#include "FEMInterpolator.h"

struct All_variables;
class Boundary;

class TractionInterpolator: public FEMInterpolator {
    const Boundary& boundary;
    const Array2D<int,1>& meshnode;
    Array2D<double,DIM> gtraction;

public:
    TractionInterpolator(const Boundary& boundary,
			 const All_variables* E,
			 Array2D<int,1>& meshNode);

    virtual ~TractionInterpolator();

    virtual void interpolateTraction(Array2D<double,DIM>& F);
    virtual void interpolateVelocity(Array2D<double,DIM>& V);

private:
    void computeTraction();
    void get_elt_traction(int el, int far, int NS,
			  int lev, int mm);
    void get_global_stress(const All_variables* E);

    // disable
    virtual void interpolateDisplacement(Array2D<double,DIM>& D) {};
    virtual void interpolateForce(Array2D<double,DIM>& F) {};
    virtual void interpolatePressure(Array2D<double,1>& P) {};
    virtual void interpolateStress(Array2D<double,DIM>& S) {};
    virtual void interpolateTemperature(Array2D<double,1>& T) {};

};



#endif

// version
// $Id: TractionInterpolator.h,v 1.5 2004/03/28 23:19:00 tan2 Exp $

// End of file
