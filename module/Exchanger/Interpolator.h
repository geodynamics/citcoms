// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_Interpolator_h)
#define pyCitcom_Interpolator_h

#include "Array2D.h"
#include "DIM.h"

struct All_variables;
class BoundedMesh;


class Interpolator {
    double theta_tol;
    double fi_tol;
    double r_tol;

    Array2D<int,1> elem_;  // elem # from which fields are interpolated
    Array2D<double,NODES_PER_ELEMENT> shape_; // shape functions for interpolation

public:
    Interpolator(const BoundedMesh& b, const All_variables* E,
		 Array2D<int,1>& meshNode);
    ~Interpolator() {};

    inline int size() const {return elem_.size();}
    void interpolateV(Array2D<double,DIM>& target,
		      const All_variables* E) const;
    void interpolateT(Array2D<double,1>& target,
		      const All_variables* E) const;

private:
    void init(const BoundedMesh& boundedMesh, const All_variables* E,
	      Array2D<int,1>& meshNode);
    void selfTest(const BoundedMesh& boundedMesh, const All_variables* E,
		  const Array2D<int,1>& meshNode) const;

    void findMaxGridSpacing(const All_variables* E);
    bool isCandidate(const double* xc, const BoundedBox& bbox) const;
    double TetrahedronVolume(double *x1, double *x2,
			     double *x3, double *x4) const;
    double det3_sub(double  *x1, double *x2, double *x3) const;
    void appendFoundElement(int el, int ntetra,
			    const double* det, double dett);

};



#endif

// version
// $Id: Interpolator.h,v 1.1 2003/11/07 01:08:01 tan2 Exp $

// End of file
