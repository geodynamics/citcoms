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

#if !defined(pyCitcomSExchanger_CitcomSource_h)
#define pyCitcomSExchanger_CitcomSource_h

#include <vector>
#include "mpi.h"
#include "Exchanger/Source.h"

struct All_variables;


class CitcomSource : public Exchanger::Source {
    const All_variables* E;

public:
    CitcomSource(MPI_Comm comm, int sink,
		 Exchanger::BoundedMesh& mesh,
		 const Exchanger::BoundedBox& mybbox,
		 const All_variables* E);
    virtual ~CitcomSource();

    virtual void interpolatePressure(Exchanger::Array2D<double,1>& P) const;
    virtual void interpolateStress(Exchanger::Array2D<double,Exchanger::STRESS_DIM>& S) const;
    virtual void interpolateTemperature(Exchanger::Array2D<double,1>& T) const;
    virtual void interpolateVelocity(Exchanger::Array2D<double,Exchanger::DIM>& V) const;

private:
    virtual void createInterpolator(const Exchanger::BoundedMesh& mesh);

    // disable these functions
    virtual void interpolateDisplacement(Exchanger::Array2D<double,Exchanger::DIM>& D) const {};
    virtual void interpolateForce(Exchanger::Array2D<double,Exchanger::DIM>& F) const {};
    virtual void interpolateHeatflux(Exchanger::Array2D<double,Exchanger::DIM>& H) const {};
    virtual void interpolateTraction(Exchanger::Array2D<double,Exchanger::DIM>& F) const {};

    // disable copy c'tor and assignment operator
    CitcomSource(const CitcomSource&);
    CitcomSource& operator=(const CitcomSource&);

};


#endif

// version
// $Id$

// End of file
