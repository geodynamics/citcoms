// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "Convertor.h"
#include "BoundedMesh.h"


BoundedMesh::BoundedMesh() :
    bbox_(DIM)
{}


BoundedMesh::~BoundedMesh()
{}


void BoundedMesh::broadcast(const MPI_Comm& comm, int broadcaster)
{
    bbox_.broadcast(comm, broadcaster);
    bbox_.print("BBox_recv");
    X_.broadcast(comm, broadcaster);
    X_.print("X_recv");

    Convertor& convertor = Convertor::instance();
    convertor.xcoordinate(bbox_);
    convertor.xcoordinate(X_);
}


void BoundedMesh::broadcast(const MPI_Comm& comm, int broadcaster) const
{
    Convertor& convertor = Convertor::instance();

    BoundedBox bbox(bbox_);
    Array2D<double,DIM> X(X_);

    convertor.coordinate(bbox);
    convertor.coordinate(X);

    bbox.broadcast(comm, broadcaster);
    X.broadcast(comm, broadcaster);
}


// version
// $Id: BoundedMesh.cc,v 1.5 2004/01/07 21:54:00 tan2 Exp $

// End of file
