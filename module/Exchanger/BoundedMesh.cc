// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "global_defs.h"
#include "BoundedMesh.h"


BoundedMesh::BoundedMesh() :
    bbox_(DIM)
{}


BoundedMesh::BoundedMesh(const All_variables* E) :
    bbox_(DIM)
{
    initBBox(E);
    bbox_.print("BBox");
}


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


void BoundedMesh::initBBox(const All_variables *E)
{
    double theta_max, theta_min;
    double fi_max, fi_min;
    double ri, ro;

    theta_max = fi_max = ro = std::numeric_limits<double>::min();
    theta_min = fi_min = ri = std::numeric_limits<double>::max();

    for(int n=1; n<=E->lmesh.nno; n++) {
	theta_max = std::max(theta_max, E->sx[1][1][n]);
	theta_min = std::min(theta_min, E->sx[1][1][n]);
	fi_max = std::max(fi_max, E->sx[1][2][n]);
	fi_min = std::min(fi_min, E->sx[1][2][n]);
	ro = std::max(ro, E->sx[1][3][n]);
	ri = std::min(ri, E->sx[1][3][n]);
    }

    bbox_[0][0] = theta_min;
    bbox_[1][0] = theta_max;
    bbox_[0][1] = fi_min;
    bbox_[1][1] = fi_max;
    bbox_[0][2] = ri;
    bbox_[1][2] = ro;
}


// version
// $Id: BoundedMesh.cc,v 1.1 2003/11/07 01:08:01 tan2 Exp $

// End of file
