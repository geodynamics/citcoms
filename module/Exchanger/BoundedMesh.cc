// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
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
}


void BoundedMesh::broadcast(const MPI_Comm& comm, int broadcaster) const
{
    bbox_.broadcast(comm, broadcaster);
    X_.broadcast(comm, broadcaster);
}


// version
// $Id: BoundedMesh.cc,v 1.2 2003/11/10 21:55:28 tan2 Exp $

// End of file
