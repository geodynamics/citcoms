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
#include "BoundedBox.h"
#include "DIM.h"

class BoundedMesh;


class FEMInterpolator {
protected:
    const All_variables* E;
    Array2D<int,1> elem_;  // elem # from which fields are interpolated
    Array2D<double,NODES_PER_ELEMENT> shape_; // shape functions for interpolation

public:
    FEMInterpolator(const BoundedMesh& boundedMesh,
		    const All_variables* E,
		    Array2D<int,1>& meshNode);
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
    void init(const BoundedMesh& boundedMesh,
	      Array2D<int,1>& meshNode);

    bool isCandidate(const double* xc, const BoundedBox& bbox) const;
    double TetrahedronVolume(double *x1, double *x2,
			     double *x3, double *x4) const;
    double det3_sub(double  *x1, double *x2, double *x3) const;
    void appendFoundElement(int el, int ntetra,
			    const double* det, double dett);

    void selfTest(const BoundedMesh& boundedMesh,
		  const Array2D<int,1>& meshNode) const;

    // disable copy c'tor and assignment operator
    FEMInterpolator(const FEMInterpolator&);
    FEMInterpolator& operator=(const FEMInterpolator&);

};



#endif

// version
// $Id: FEMInterpolator.h,v 1.3 2004/01/08 20:42:56 tan2 Exp $

// End of file
