// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <algorithm>
#include <limits>
#include <vector>
#include "global_defs.h"
#include "journal/journal.h"
#include "Boundary.h"


Boundary::Boundary() :
    BoundedMesh()
{}


Boundary::Boundary(const All_variables* E) :
    BoundedMesh()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Boundary::Boundary" << journal::end;

    initBBox(E);
    bbox_.print("Boundary-BBox");

    // boundary = all - interior
    int maxNodes = E->lmesh.nno - (E->lmesh.nox-2)
	                         *(E->lmesh.noy-2)
	                         *(E->lmesh.noz-2);
    X_.reserve(maxNodes);
    meshID_.reserve(maxNodes);

    initX(E);

    X_.shrink();
    X_.print("Boundary-X");
    meshID_.shrink();
    meshID_.print("Boundary-meshID");
}


void Boundary::initBBox(const All_variables *E)
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


void Boundary::initX(const All_variables* E)
{
    std::vector<int> nid(E->lmesh.nno, 0);

    //  for two XOZ planes
    
    if (E->parallel.me_loc[2] == 0)
	for (int m=1;m<=E->sphere.caps_per_proc;m++)
	    for(int j=1;j<=E->lmesh.nox;j++)
		for(int i=1;i<=E->lmesh.noz;i++)  {
		    int node1 = i + (j-1)*E->lmesh.noz;
		    
		    if (!nid[node1-1]) {
			appendX(E, m, node1);
			nid[node1-1]++;
		    }
		}

    if (E->parallel.me_loc[2] == E->parallel.nprocy-1)
	for (int m=1;m<=E->sphere.caps_per_proc;m++)
	    for(int j=1;j<=E->lmesh.nox;j++)
		for(int i=1;i<=E->lmesh.noz;i++)  {
		    int node1 = i + (j-1)*E->lmesh.noz;
		    int node2 = node1 + (E->lmesh.noy-1)*E->lmesh.noz*E->lmesh.nox;
		    if (!nid[node2-1]) {
			appendX(E, m, node2);
			nid[node2-1]++;
		    }
		}
    
    //  for two YOZ planes
    
    if (E->parallel.me_loc[1] == 0)
	for (int m=1; m<=E->sphere.caps_per_proc; m++)
	    for(int j=1; j<=E->lmesh.noy; j++)
		for(int i=1; i<=E->lmesh.noz; i++) {
		    int node1 = i + (j-1)*E->lmesh.noz*E->lmesh.nox;
		    
		    if (!nid[node1-1]) {
			appendX(E, m, node1);
			nid[node1-1]++;
		    }
		}
    
    if (E->parallel.me_loc[1] == E->parallel.nprocx-1)
	for (int m=1; m<=E->sphere.caps_per_proc; m++)
	    for(int j=1; j<=E->lmesh.noy; j++)
		for(int i=1; i<=E->lmesh.noz; i++) {
		    int node1 = i + (j-1)*E->lmesh.noz*E->lmesh.nox;
		    int node2 = node1 + (E->lmesh.nox-1)*E->lmesh.noz;
		    
		    if (!nid[node2-1]) {
			appendX(E, m, node2);
			nid[node2-1]++;
		    }
		}
    
    //  for two XOY planes
    
    if (E->parallel.me_loc[3] == 0)
	for (int m=1;m<=E->sphere.caps_per_proc;m++)
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int i=1;i<=E->lmesh.nox;i++)  {
		    int node1 = 1 + (i-1)*E->lmesh.noz+(j-1)*E->lmesh.nox*E->lmesh.noz;
		    if (!nid[node1-1]) {
			appendX(E, m, node1);
			nid[node1-1]++;
		    }
		}
    
    if (E->parallel.me_loc[3] == E->parallel.nprocz-1)
	for (int m=1;m<=E->sphere.caps_per_proc;m++)
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int i=1;i<=E->lmesh.nox;i++)  {
		    int node1 = 1 + (i-1)*E->lmesh.noz+(j-1)*E->lmesh.nox*E->lmesh.noz;
		    int node2 = node1 + E->lmesh.noz-1;
		    
		    if (!nid[node2-1]) {
			appendX(E, m, node2);
			nid[node2-1]++;
		    }
		}
}


void Boundary::appendX(const All_variables *E, int m, int node)
{
    std::vector<double> x(DIM);
    for(int k=0; k<DIM; k++)
	x[k] = E->sx[m][k+1][node];
    X_.push_back(x);
    meshID_.push_back(node);
}


// version
// $Id: Boundary.cc,v 1.40 2003/11/10 21:55:28 tan2 Exp $

// End of file
