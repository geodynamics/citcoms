// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_CitcomInterpolator_h)
#define pyCitcomSExchanger_CitcomInterpolator_h

#include "Exchanger/Interpolator.h"


class CitcomInterpolator : public Exchanger::Interpolator {
protected:
    const All_variables* E;

public:
    CitcomInterpolator(const Exchanger::BoundedMesh& boundedMesh,
		       Exchanger::Array2D<int,1>& meshNode,
		       const All_variables* E);
    virtual ~CitcomInterpolator();

    virtual void interpolatePressure(Exchanger::Array2D<double,1>& P);
    virtual void interpolateStress(Exchanger::Array2D<double,Exchanger::STRESS_DIM>& S);
    virtual void interpolateTemperature(Exchanger::Array2D<double,1>& T);
    virtual void interpolateVelocity(Exchanger::Array2D<double,Exchanger::DIM>& V);

private:
    void init(const Exchanger::BoundedMesh& boundedMesh,
	      Exchanger::Array2D<int,1>& meshNode);
    void computeElementGeometry(Exchanger::Array2D<double,Exchanger::DIM*Exchanger::DIM>& etaAxes,
				Exchanger::Array2D<double,Exchanger::DIM>& inv_length_sq) const;
    int bisectInsertPoint(double x, const std::vector<double>& v) const;
    bool elementInverseMapping(std::vector<double>& elmShape,
			       const std::vector<double>& x,
			       const Exchanger::Array2D<double,Exchanger::DIM*Exchanger::DIM>& etaAxes,
			       const Exchanger::Array2D<double,Exchanger::DIM>& inv_length_sq,
			       int element, double accuracy);
    void getShapeFunction(std::vector<double>& shape,
			  const std::vector<double>& eta) const;
    void selfTest(const Exchanger::BoundedMesh& boundedMesh,
		  const Exchanger::Array2D<int,1>& meshNode) const;

    // diable
    virtual void interpolateDisplacement(Exchanger::Array2D<double,Exchanger::DIM>& D) {};
    virtual void interpolateForce(Exchanger::Array2D<double,Exchanger::DIM>& F) {};
    virtual void interpolateHeatflux(Exchanger::Array2D<double,Exchanger::DIM>& H) {};
    virtual void interpolateTraction(Exchanger::Array2D<double,Exchanger::DIM>& F) {};

    // disable copy c'tor and assignment operator
    CitcomInterpolator(const CitcomInterpolator&);
    CitcomInterpolator& operator=(const CitcomInterpolator&);

};


#endif

// version
// $Id: CitcomInterpolator.h,v 1.1 2004/05/11 07:55:30 tan2 Exp $

// End of file
