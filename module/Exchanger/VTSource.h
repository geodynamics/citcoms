// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_VTSource_h)
#define pyCitcom_VTSource_h

#include <vector>
#include "mpi.h"
#include "utility.h"
#include "AbstractSource.h"
#include "Array2D.h"
#include "BoundedBox.h"
#include "DIM.h"

struct All_variables;
class BoundedMesh;
class VTInterpolator;


class VTSource : public AbstractSource {
    const All_variables* E;

public:
    VTSource(MPI_Comm comm, int sink,
	     BoundedMesh& mesh, const All_variables* E,
	     const BoundedBox& mybbox);
    virtual ~VTSource();

    virtual void interpolateStress(Array2D<double,STRESS_DIM>& S) const;
    virtual void interpolateTemperature(Array2D<double,1>& T) const;
    virtual void interpolateVelocity(Array2D<double,DIM>& V) const;

private:
    virtual void createInterpolator(const BoundedMesh& mesh);

    // disable these functions
    virtual void interpolateDisplacement(Array2D<double,DIM>& D) const {};
    virtual void interpolateForce(Array2D<double,DIM>& F) const {};
    virtual void interpolatePressure(Array2D<double,1>& P) const {};
    virtual void interpolateTraction(Array2D<double,DIM>& F) const {};

    // disable copy c'tor and assignment operator
    VTSource(const VTSource&);
    VTSource& operator=(const VTSource&);

};


#endif

// version
// $Id: VTSource.h,v 1.4 2004/04/14 20:12:13 tan2 Exp $

// End of file
