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
    BoundedMesh(E)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in Boundary::Boundary" << journal::end;

    // boundary = all - interior
    int maxNodes = E->lmesh.nno - (E->lmesh.nox-2)
	                         *(E->lmesh.noy-2)
	                         *(E->lmesh.noz-2);
    X_.reserve(maxNodes);
    meshID_.reserve(maxNodes);

    initX(E);

    X_.shrink();
    X_.print("X");
    meshID_.shrink();
    meshID_.print("meshID");
}


void Boundary::initX(const All_variables* E)
{
    std::vector<int> nid(E->lmesh.nno, 0);

    //  for two YOZ planes

    if (E->parallel.me_loc[1] == 0 ||
	E->parallel.me_loc[1] == E->parallel.nprocx-1)
	for (int m=1; m<=E->sphere.caps_per_proc; m++)
	    for(int j=1; j<=E->lmesh.noy; j++)
		for(int i=1; i<=E->lmesh.noz; i++)  {
		    int node1 = i + (j-1)*E->lmesh.noz*E->lmesh.nox;
		    int node2 = node1 + (E->lmesh.nox-1)*E->lmesh.noz;

		    if ((E->parallel.me_loc[1]==0) && (!nid[node1-1]))  {
			appendX(E, m, node1);
			nid[node1-1]++;
		    }
		    if ((E->parallel.me_loc[1]==E->parallel.nprocx-1) && (!nid[node2-1])) {
			appendX(E, m, node2);
			nid[node2-1]++;
		    }
		}

    //  for two XOZ planes

    if (E->parallel.me_loc[2]==0 || E->parallel.me_loc[2]==E->parallel.nprocy-1)
	for (int m=1;m<=E->sphere.caps_per_proc;m++)
	    for(int j=1;j<=E->lmesh.nox;j++)
		for(int i=1;i<=E->lmesh.noz;i++)  {
		    int node1 = i + (j-1)*E->lmesh.noz;
		    int node2 = node1 + (E->lmesh.noy-1)*E->lmesh.noz*E->lmesh.nox;
		    if ((E->parallel.me_loc[2]==0) && (!nid[node1-1]))  {
			appendX(E, m, node1);
			nid[node1-1]++;
		    }
		    if((E->parallel.me_loc[2]==E->parallel.nprocy-1)&& (!nid[node2-1]))  {
			appendX(E, m, node2);
			nid[node2-1]++;
		    }
		}
    //  for two XOY planes
    if (E->parallel.me_loc[3]==0 || E->parallel.me_loc[3]==E->parallel.nprocz-1)
	for (int m=1;m<=E->sphere.caps_per_proc;m++)
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int i=1;i<=E->lmesh.nox;i++)  {
		    int node1 = 1 + (i-1)*E->lmesh.noz+(j-1)*E->lmesh.nox*E->lmesh.noz;
		    int node2 = node1 + E->lmesh.noz-1;

		    if ((E->parallel.me_loc[3]==0 ) && (!nid[node1-1])) {
			appendX(E, m, node1);
			nid[node1-1]++;
		    }
		    if ((E->parallel.me_loc[3]==E->parallel.nprocz-1) &&(!nid[node2-1])) {
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
// $Id: Boundary.cc,v 1.39 2003/11/07 01:08:01 tan2 Exp $

// End of file
