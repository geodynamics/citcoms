// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
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
// $Id: CitcomSource.h,v 1.1 2004/05/11 07:55:30 tan2 Exp $

// End of file
