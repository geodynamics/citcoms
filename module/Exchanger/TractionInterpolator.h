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
class BoundedMesh;

class TractionInterpolator: public FEMInterpolator {
    Array2D<float,DIM> gtraction;

    int xmin;
    int xmax;
    int ymin;
    int ymax;
    int zmin;
    int zmax;

    int dm_xmin;
    int dm_xmax;
    int dm_ymin;
    int dm_ymax;
    int dm_zmin;
    int dm_zmax;

    int do_xmin;
    int do_xmax;
    int do_ymin;
    int do_ymax;
    int do_zmin;
    int do_zmax;

public:
    TractionInterpolator(const BoundedMesh& boundedMesh,
			 const All_variables* E,
			 Array2D<int,1>& meshNode);

    virtual ~TractionInterpolator();

    virtual void interpolateTraction(Array2D<double,DIM>& F);
    void domain_cutout();

private:
    void initComputeTraction(const BoundedMesh& boundedMesh);
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
    virtual void interpolateVelocity(Array2D<double,DIM>& V) {};

};



#endif

// version
// $Id: TractionInterpolator.h,v 1.4 2004/01/15 03:00:24 ces74 Exp $

// End of file
