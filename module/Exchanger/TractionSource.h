// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_TractionSource_h)
#define pyCitcom_TractionSource_h

#include "mpi.h"
#include "utility.h"
#include "AbstractSource.h"
#include "Array2D.h"

struct All_variables;
class Boundary;


class TractionSource : public AbstractSource {
    const All_variables* E;

public:
    TractionSource(MPI_Comm comm, int sinkRank,
		   Boundary& mesh, const All_variables* E,
		   const BoundedBox& mybbox);
    virtual ~TractionSource();

    virtual void interpolateTraction(Array2D<double,DIM>& F) const;
    virtual void interpolateVelocity(Array2D<double,DIM>& V) const;

private:
    virtual void createInterpolator(const BoundedMesh& mesh);

    // disable these functions
    virtual void interpolateDisplacement(Array2D<double,DIM>& D) const {};
    virtual void interpolateForce(Array2D<double,DIM>& F) const {};
    virtual void interpolatePressure(Array2D<double,1>& P) const {};
    virtual void interpolateStress(Array2D<double,STRESS_DIM>& S) const {};
    virtual void interpolateTemperature(Array2D<double,1>& T) const {};

};


#endif

// version
// $Id: TractionSource.h,v 1.6 2004/04/14 19:41:38 tan2 Exp $

// End of file
