// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "BoundedBox.h"
#include "BoundedMesh.h"
#include "journal/journal.h"
#include "VTInterpolator.h"
#include "Source.h"


Source::Source(MPI_Comm c, int s,
	       BoundedMesh& mesh, const All_variables* E,
	       const BoundedBox& mybbox) :
    comm(c),
    sink(s)
{
    recvMesh(mesh);

    if(isOverlapped(mesh.bbox(), mybbox)) {
  	interp.reset(new VTInterpolator(mesh, E, meshNode_));
	meshNode_.print("meshNode");
    }
    sendMeshNode();
    initX(mesh);
}


Source::~Source()
{}


void Source::interpolateT(Array2D<double,1>& T, const All_variables* E) const
{
    journal::debug_t debug("Citcom_Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Source::interpolateT" << journal::end;

    if(size())
	interp->interpolateTemperature(T);
}


void Source::interpolateV(Array2D<double,DIM>& V, const All_variables* E) const
{
    if(size())
	interp->interpolateVelocity(V);
}


// private functions

void Source::recvMesh(BoundedMesh& mesh)
{
    // assuming sink is broadcasting mesh to every source
    mesh.broadcast(comm, sink);
}


void Source::sendMeshNode() const
{
    meshNode_.sendSize(comm, sink);
    sendArray2D(meshNode_);
}


void Source::initX(const BoundedMesh& mesh)
{
    X_.resize(meshNode_.size());

    for(int i=0; i<X_.size(); ++i) {
	int n = meshNode_[0][i];
	for(int j=0; j<DIM; ++j)
	    X_[j][i] = mesh.X(j,n);
    }
}


// version
// $Id: Source.cc,v 1.9 2004/01/14 00:34:02 tan2 Exp $

// End of file
