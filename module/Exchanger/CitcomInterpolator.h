// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//
// CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
// Copyright (C) 2002-2005, California Institute of Technology.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//</LicenseText>
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
// $Id$

// End of file
