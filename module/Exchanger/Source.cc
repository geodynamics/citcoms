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
#include "Interpolator.h"
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


void Source::interpolateF(Array2D<double,DIM>& F, const All_variables* E) const
{
    if(size())
	interp->interpolateF(F, E);
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



// version
// $Id: Source.cc,v 1.3 2003/11/23 18:11:08 ces74 Exp $

// End of file
