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

    virtual ~TractionInterpolator() {};

    void InterpolateTraction(Array2D<double,DIM>& F,All_variables* E);
    virtual void domain_cutout(const All_variables* E);
private:
    void init(const BoundedMesh& boundedMesh,
	      const All_variables* E,
	      Array2D<int,1>& meshNode);

    void initComputeTraction(const BoundedMesh& boundedMesh,
			     const All_variables* E);
    void computeTraction(All_variables* E);
    void get_elt_traction(All_variables* E,int el,int far,int NS,int lev,int mm);

    void findMaxGridSpacing(const All_variables* E);
    bool isCandidate(const double* xc, const BoundedBox& bbox) const;
    double TetrahedronVolume(double *x1, double *x2,
			     double *x3, double *x4) const;
    double det3_sub(double  *x1, double *x2, double *x3) const;
    void appendFoundElement(int el, int ntetra,
			    const double* det, double dett);

    void selfTest(const BoundedMesh& boundedMesh,
		  const Array2D<int,1>& meshNode) const;

    virtual void interpolateDisplacement(Array2D<double,DIM>& D) const {};
    virtual void interpolateForce(Array2D<double,DIM>& F) const {};
    virtual void interpolatePressure(Array2D<double,1>& P) const {};
    virtual void interpolateStress(Array2D<double,DIM>& S) const {};
    virtual void interpolateTemperature(Array2D<double,1>& T) const {};
    virtual void interpolateTraction(Array2D<double,DIM>& T) const {};
    virtual void interpolateVelocity(Array2D<double,DIM>& V) const {};
};



#endif

// version
// $Id: TractionInterpolator.h,v 1.1 2003/11/28 22:34:16 ces74 Exp $

// End of file
