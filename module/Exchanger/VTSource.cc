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
#include "BoundedMesh.h"
#include "VTInterpolator.h"
#include "VTSource.h"

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


VTSource::VTSource(MPI_Comm comm, int sink,
		   BoundedMesh& mesh, const All_variables* e,
		   const BoundedBox& mybbox) :
    AbstractSource(comm, sink, mesh, mybbox),
    E(e)
{
    init(mesh, mybbox);
}


VTSource::~VTSource()
{}


void VTSource::interpolateStress(Array2D<double,STRESS_DIM>& S) const
{
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


void VTSource::interpolateTemperature(Array2D<double,1>& T) const
{
    if(size())
	interp->interpolateTemperature(T);
}


void VTSource::interpolateVelocity(Array2D<double,DIM>& V) const
{
    if(size())
	interp->interpolateVelocity(V);
}


// private functions

void VTSource::createInterpolator(const BoundedMesh& mesh)
{
    interp = new VTInterpolator(mesh, E, meshNode_);
}


// version
// $Id: VTSource.cc,v 1.3 2004/04/14 20:12:13 tan2 Exp $

// End of file
