// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "journal/journal.h"
#include "BoundedBox.h"
#include "BoundedMesh.h"
#include "VTInterpolator.h"
#include "VTSource.h"


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
// $Id: VTSource.cc,v 1.2 2004/02/25 23:07:35 tan2 Exp $

// End of file
