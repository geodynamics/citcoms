// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "global_defs.h"
#include "journal/journal.h"
#include "BoundedBox.h"
#include "Boundary.h"
#include "TractionInterpolator.h"
#include "TractionSource.h"

extern "C" {
    void allocate_STD_mem(const struct All_variables *E,
			  float** SXX, float** SYY, float** SZZ,
			  float** SXY, float** SXZ, float** SZY,
			  float** divv, float** vorv);
    void free_STD_mem(const struct All_variables *E,
		      float** SXX, float** SYY, float** SZZ,
		      float** SXY, float** SXZ, float** SZY,
		      float** divv, float** vorv);
    void compute_nodal_stress(const struct All_variables *E,
			      float** SXX, float** SYY, float** SZZ,
			      float** SXY, float** SXZ, float** SZY,
			      float** divv, float** vorv);
}


TractionSource::TractionSource(MPI_Comm comm, int sink,
			       Boundary& mesh, const All_variables* e,
			       const BoundedBox& mybbox) :
    AbstractSource(comm, sink, mesh, mybbox),
    E(e)
{
    init(mesh, mybbox);
}


TractionSource::~TractionSource()
{}


void TractionSource::interpolateTraction(Array2D<double,DIM>& F) const
{
    // copied and modified from get_STD_topo() in Topo_gravity.c

    float *SXX[NCS],*SYY[NCS],*SXY[NCS],*SXZ[NCS],*SZY[NCS],*SZZ[NCS];
    float *divv[NCS],*vorv[NCS];

    allocate_STD_mem(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);
    compute_nodal_stress(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);
    free_STD_mem(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);

    if(size())
	interp->interpolateTraction(F);
}


void TractionSource::interpolateVelocity(Array2D<double,DIM>& V) const
{
    if(size())
	interp->interpolateVelocity(V);
}


// private functions

void TractionSource::createInterpolator(const BoundedMesh& mesh)
{
    interp = new TractionInterpolator(dynamic_cast<const Boundary&>(mesh),
				      E, meshNode_);
}


// version
// $Id: TractionSource.cc,v 1.6 2004/03/28 23:19:00 tan2 Exp $

// End of file
