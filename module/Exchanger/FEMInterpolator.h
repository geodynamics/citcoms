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
    virtual void interpolateStress(Array2D<double,STRESS_DIM>& S) = 0;
    virtual void interpolateTemperature(Array2D<double,1>& T) = 0;
    virtual void interpolateTraction(Array2D<double,DIM>& F) = 0;
    virtual void interpolateVelocity(Array2D<double,DIM>& V) = 0;

private:
    void init(const BoundedMesh& boundedMesh,
	      Array2D<int,1>& meshNode);
    void computeElementGeometry(Array2D<double,DIM*DIM>& etaAxes,
				Array2D<double,DIM>& inv_length_sq) const;
    int bisectInsertPoint(double x, const std::vector<double>& v) const;
    bool elementInverseMapping(std::vector<double>& elmShape,
		       const std::vector<double>& x,
		       const Array2D<double,DIM*DIM>& etaAxes,
		       const Array2D<double,DIM>& inv_length_sq,
		       int element, double accuracy);
    void getShapeFunction(std::vector<double>& shape,
			  const std::vector<double>& eta) const;
    void selfTest(const BoundedMesh& boundedMesh,
		  const Array2D<int,1>& meshNode) const;

    // disable copy c'tor and assignment operator
    FEMInterpolator(const FEMInterpolator&);
    FEMInterpolator& operator=(const FEMInterpolator&);

};



#endif

// version
// $Id: FEMInterpolator.h,v 1.5 2004/04/14 19:41:38 tan2 Exp $

// End of file
