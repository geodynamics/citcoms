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
#include "Exchanger/BoundedMesh.h"
#include "CitcomInterpolator.h"

using Exchanger::Array2D;
using Exchanger::BoundedBox;
using Exchanger::BoundedMesh;
using Exchanger::DIM;
using Exchanger::STRESS_DIM;

#include "CitcomSource.h"

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


CitcomSource::CitcomSource(MPI_Comm comm,
			   int sinkRank,
			   BoundedMesh& mesh,
			   const BoundedBox& mybbox,
			   const All_variables* e) :
    Exchanger::Source(comm, sinkRank, mesh, mybbox),
    E(e)
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    init(mesh, mybbox);
}


CitcomSource::~CitcomSource()
{}


void CitcomSource::interpolatePressure(Array2D<double,1>& P) const
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    if(size())
	interp->interpolatePressure(P);
}


void CitcomSource::interpolateStress(Array2D<double,STRESS_DIM>& S) const
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;
    // copied and modified from get_STD_topo() in Topo_gravity.c

    float *SXX[NCS], *SYY[NCS], *SZZ[NCS];
    float *SXY[NCS], *SXZ[NCS], *SZY[NCS];
    float *divv[NCS], *vorv[NCS];

    allocate_STD_mem(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);
    compute_nodal_stress(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);
    free_STD_mem(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);

    if(size())
	interp->interpolateStress(S);
}


void CitcomSource::interpolateTemperature(Array2D<double,1>& T) const
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    if(size())
	interp->interpolateTemperature(T);
}


void CitcomSource::interpolateVelocity(Array2D<double,DIM>& V) const
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    if(size())
	interp->interpolateVelocity(V);
}


// private functions

void CitcomSource::createInterpolator(const BoundedMesh& mesh)
{
    interp = new CitcomInterpolator(mesh, meshNode_, E);
}


// version
// $Id: CitcomSource.cc,v 1.1 2004/05/11 07:55:30 tan2 Exp $

// End of file
