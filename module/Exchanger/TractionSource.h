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

public:
    TractionSource(MPI_Comm comm, int sinkRank,
		   BoundedMesh& mesh, const All_variables* E,
		   const BoundedBox& mybbox);
    virtual ~TractionSource() {};

    virtual void interpolateTraction(Array2D<double,DIM>& F) const;
    void domain_cutout();

private:
    void recvMesh(BoundedMesh& mesh);
    void sendMeshNode() const;
    void initX(const BoundedMesh& mesh);

    // disable these functions
    virtual void interpolateForce(Array2D<double,DIM>& F) const {};
    virtual void interpolatePressure(Array2D<double,1>& P) const {};
    virtual void interpolateStress(Array2D<double,DIM>& S) const {};
    virtual void interpolateTemperature(Array2D<double,1>& T) const {};
    virtual void interpolateVelocity(Array2D<double,DIM>& V) const {};
};


#endif

// version
// $Id: TractionSource.h,v 1.3 2004/01/14 02:11:24 ces74 Exp $

// End of file
