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
#include "Source.h"


Source::Source(MPI_Comm c, int s,
	       BoundedMesh& mesh, const All_variables* E,
	       const BoundedBox& mybbox) :
    comm(c),
    sink(s)
{
    recvMesh(mesh);

    if(isOverlapped(mesh.bbox(), mybbox)) {
  	interp.reset(new Interpolator(mesh, E, meshNode_));
	meshNode_.print("meshNode");
    }
    sendMeshNode();
    initX(mesh);
}


void Source::interpolateT(Array2D<double,1>& T, const All_variables* E) const
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Source::interpolateT" << journal::end;

    if(size())
	interp->interpolateT(T, E);
}


void Source::interpolateV(Array2D<double,DIM>& V, const All_variables* E) const
{
    if(size())
	interp->interpolateV(V, E);
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
// $Id: Source.cc,v 1.6 2004/01/08 02:29:37 tan2 Exp $

// End of file
