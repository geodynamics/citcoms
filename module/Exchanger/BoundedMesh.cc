// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "Dimensional.h"
#include "BoundedMesh.h"


BoundedMesh::BoundedMesh(bool dimensional) :
    bbox_(DIM),
    dimensional_(dimensional)
{}


BoundedMesh::~BoundedMesh()
{}


void BoundedMesh::broadcast(const MPI_Comm& comm, int broadcaster)
{
    bbox_.broadcast(comm, broadcaster);
    bbox_.print("BBox_recv");
    X_.broadcast(comm, broadcaster);
    X_.print("X_recv");

    if(dimensional_) {
	Dimensional& dimen = Dimensional::instance();
	dimen.xcoordinate(bbox_);
	dimen.xcoordinate(X_);
    }
}


void BoundedMesh::broadcast(const MPI_Comm& comm, int broadcaster) const
{
    if(dimensional_) {
	Dimensional& dimen = Dimensional::instance();

	BoundedBox bbox(bbox_);
	dimen.coordinate(bbox);

	Array2D<double,DIM> X(X_);
	dimen.coordinate(X);

	bbox.broadcast(comm, broadcaster);
	X.broadcast(comm, broadcaster);
    }
    else {
	bbox_.broadcast(comm, broadcaster);
	X_.broadcast(comm, broadcaster);
    }
}


// version
// $Id: BoundedMesh.cc,v 1.3 2003/12/30 21:46:01 tan2 Exp $

// End of file
