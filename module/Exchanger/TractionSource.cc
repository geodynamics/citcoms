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
#include "TractionInterpolator.h"
#include "TractionSource.h"


TractionSource::TractionSource(MPI_Comm c, int s,
			       BoundedMesh& mesh, const All_variables* E,
			       const BoundedBox& mybbox) :
    AbstractSource(c, s)
{
    recvMesh(mesh);

    if(isOverlapped(mesh.bbox(), mybbox)) {
	interp = new TractionInterpolator(mesh, meshNode_, E);
	meshNode_.print("meshNode");
    }
    sendMeshNode();
    initX(mesh);
}


void TractionSource::interpolateTraction(Array2D<double,DIM>& F)
{
    if(size())
	interp->interpolateTraction(F);
}


void TractionSource::domain_cutout()
{
    (dynamic_cast<TractionInterpolator*>(interp))->domain_cutout();
}


// private functions

void TractionSource::recvMesh(BoundedMesh& mesh)
{
    // assuming sink is broadcasting mesh to every source
    mesh.broadcast(comm, sink);
}


void TractionSource::sendMeshNode() const
{
    meshNode_.sendSize(comm, sink);
    send(meshNode_);
}


void TractionSource::initX(const BoundedMesh& mesh)
{
    X_.resize(meshNode_.size());

    for(int i=0; i<X_.size(); ++i) {
	int n = meshNode_[0][i];
	for(int j=0; j<DIM; ++j)
	    X_[j][i] = mesh.X(j,n);
    }
}


// version
// $Id: TractionSource.cc,v 1.2 2004/01/08 02:29:37 tan2 Exp $

// End of file
