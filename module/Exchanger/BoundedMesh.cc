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
#include "Transformational.h"
#include "BoundedMesh.h"


BoundedMesh::BoundedMesh(bool dimensional, bool transformational) :
    bbox_(DIM),
    dimensional_(dimensional),
    transformational_(transformational)
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
    if(transformational_){
 //         Transformational& trans = Transformational::instance();
        
//          trans.xcoordinate(bbox_);
//          trans.xcoordinate(X_);
    }        
}


void BoundedMesh::broadcast(const MPI_Comm& comm, int broadcaster) const
{
    if(dimensional_ || transformational_) {
	Dimensional& dimen = Dimensional::instance();
//        if(transformational_) Transformational& trans = Transformational::instance();
        
	BoundedBox bbox(bbox_);
//        if(transformational_) trans.coordinate(bbox);
        
	Array2D<double,DIM> X(X_);
//        if(transformational_) trans.coordinate(X);
        
	bbox.broadcast(comm, broadcaster);
	X.broadcast(comm, broadcaster);
    }
    else {
	bbox_.broadcast(comm, broadcaster);
	X_.broadcast(comm, broadcaster);
    }
}


// version
// $Id: BoundedMesh.cc,v 1.4 2004/01/06 22:40:28 puru Exp $

// End of file
