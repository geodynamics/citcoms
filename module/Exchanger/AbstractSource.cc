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
#include "BoundedMesh.h"
#include "FEMInterpolator.h"
#include "AbstractSource.h"


AbstractSource::AbstractSource(MPI_Comm c, int sinkRank,
			       BoundedMesh& mesh, const BoundedBox& mybbox) :
    comm(c),
    sink(sinkRank)
{}


AbstractSource::~AbstractSource()
{
    delete interp;
}


// protected functions

void AbstractSource::init(BoundedMesh& mesh, const BoundedBox& mybbox)
{
    recvMesh(mesh, mybbox);
    if(isOverlapped(mesh.bbox(), mybbox)) {
	createInterpolator(mesh);
        meshNode_.print("meshNode");
    }
    sendMeshNode();
    initX(mesh);
}


void AbstractSource::recvMesh(BoundedMesh& mesh, const BoundedBox& mybbox)
{
#if 1
    // assuming sink is broadcasting mesh to every source
    mesh.broadcast(comm, sink);

#else
    BoundedBox bbox = mybbox;
    util::exchange(comm, sink, bbox);

    if(isOverlapped(mybbox, bbox)) {
	mesh.receive(comm, source[i]);
	createInterpolator(mesh);
    }

#endif
}


void AbstractSource::sendMeshNode() const
{
    meshNode_.sendSize(comm, sink);
    send(meshNode_);
}


void AbstractSource::initX(const BoundedMesh& mesh)
{
    X_.resize(meshNode_.size());

    for(int i=0; i<X_.size(); ++i) {
	int n = meshNode_[0][i];
	for(int j=0; j<DIM; ++j)
	    X_[j][i] = mesh.X(j,n);
    }
}


// version
// $Id: AbstractSource.cc,v 1.1 2004/02/25 23:07:35 tan2 Exp $

// End of file
