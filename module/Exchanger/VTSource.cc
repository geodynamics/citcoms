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
		   BoundedMesh& mesh, const All_variables* E,
		   const BoundedBox& mybbox) :
    AbstractSource(comm, sink)
{
    recvMesh(mesh);

    if(isOverlapped(mesh.bbox(), mybbox)) {
  	interp = new VTInterpolator(mesh, E, meshNode_);
	meshNode_.print("meshNode");
    }
    sendMeshNode();
    initX(mesh);
}


VTSource::~VTSource()
{}


void VTSource::interpolateTemperature(Array2D<double,1>& T) const
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    if(size())
	interp->interpolateTemperature(T);
}


void VTSource::interpolateVelocity(Array2D<double,DIM>& V) const
{
    if(size())
	interp->interpolateVelocity(V);
}


// private functions

void VTSource::recvMesh(BoundedMesh& mesh)
{
    // assuming sink is broadcasting mesh to every source
    mesh.broadcast(comm, sink);
}


void VTSource::sendMeshNode() const
{
    meshNode_.sendSize(comm, sink);
    send(meshNode_);
}


void VTSource::initX(const BoundedMesh& mesh)
{
    X_.resize(meshNode_.size());

    for(int i=0; i<X_.size(); ++i) {
	int n = meshNode_[0][i];
	for(int j=0; j<DIM; ++j)
	    X_[j][i] = mesh.X(j,n);
    }
}


// version
// $Id: VTSource.cc,v 1.1 2004/02/24 20:20:32 tan2 Exp $

// End of file
