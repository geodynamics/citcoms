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


// version
// $Id: TractionSource.cc,v 1.1 2003/12/16 18:50:53 tan2 Exp $

// End of file
