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
class BoundedMesh;


class TractionSource : public AbstractSource {
    const All_variables* E;

public:
    TractionSource(MPI_Comm comm, int sinkRank,
		   BoundedMesh& mesh, const All_variables* E,
		   const BoundedBox& mybbox);
    virtual ~TractionSource();

    virtual void interpolateTraction(Array2D<double,DIM>& F) const;
    void domain_cutout();

private:
    virtual void createInterpolator(const BoundedMesh& mesh);

    // disable these functions
    virtual void interpolateForce(Array2D<double,DIM>& F) const {};
    virtual void interpolatePressure(Array2D<double,1>& P) const {};
    virtual void interpolateStress(Array2D<double,DIM>& S) const {};
    virtual void interpolateTemperature(Array2D<double,1>& T) const {};
    virtual void interpolateVelocity(Array2D<double,DIM>& V) const {};
};


#endif

// version
// $Id: TractionSource.h,v 1.4 2004/02/25 23:07:35 tan2 Exp $

// End of file
