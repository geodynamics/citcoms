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
    const All_variables* E;
    float *gtraction[DIM];

    double theta_tol;
    double fi_tol;
    double r_tol;

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
			 Array2D<int,1>& meshNode,
			 const All_variables* E);

    virtual ~TractionInterpolator();

    virtual void InterpolateTraction(Array2D<double,DIM>& F);
    void domain_cutout();

private:
    void init(const BoundedMesh& boundedMesh,
	      Array2D<int,1>& meshNode);

    void initComputeTraction(const BoundedMesh& boundedMesh);
    void computeTraction();
    void get_elt_traction(int el, int far, int NS,
			  int lev, int mm);

    void findMaxGridSpacing();
    bool isCandidate(const double* xc, const BoundedBox& bbox) const;
    double TetrahedronVolume(double *x1, double *x2,
			     double *x3, double *x4) const;
    double det3_sub(double  *x1, double *x2, double *x3) const;
    void appendFoundElement(int el, int ntetra,
			    const double* det, double dett);

    void selfTest(const BoundedMesh& boundedMesh,
		  const Array2D<int,1>& meshNode) const;

    // disable
    virtual void interpolateDisplacement(Array2D<double,DIM>& D) {};
    virtual void interpolateForce(Array2D<double,DIM>& F) {};
    virtual void interpolatePressure(Array2D<double,1>& P) {};
    virtual void interpolateStress(Array2D<double,DIM>& S) {};
    virtual void interpolateTemperature(Array2D<double,1>& T) {};
    virtual void interpolateTraction(Array2D<double,DIM>& T) {};
    virtual void interpolateVelocity(Array2D<double,DIM>& V) {};
};



#endif

// version
// $Id: TractionInterpolator.h,v 1.2 2003/12/16 02:14:10 tan2 Exp $

// End of file
