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
#include "BoundedBox.h"
#include "Interior.h"


Interior::Interior() :
    BoundedMesh()
{}



Interior::Interior(const BoundedBox& remoteBBox, const All_variables* E) :
    BoundedMesh(E)
{

#if 0
    int node,n,l;

    int size_=0;
    for (int m=1;m<=E->sphere.caps_per_proc;m++)
        for(int i=1;i<=E->lmesh.nox;i++)
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int k=1;k<=E->lmesh.noz;k++)
                {
                    node = k + (i-1)*E->lmesh.noz+(j-1)*E->lmesh.nox*E->lmesh.noz;
                    if((E->sx[m][1][node]> b->theta_min()) &&
                       (E->sx[m][1][node]< b->theta_max()) &&
                       (E->sx[m][2][node]> b->fi_min()) &&
                       (E->sx[m][2][node]< b->fi_max()) &&
                       (E->sx[m][3][node]> b->ri()) &&
                       (E->sx[m][3][node]< b->ro()))
                    {
                        size_++;
                    }

                }
    X_.resize(size_);
     n=0;
    for (int m=1;m<=E->sphere.caps_per_proc;m++)
        for(int i=1;i<=E->lmesh.nox;i++)
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int k=1;k<=E->lmesh.noz;k++)
                {
                    node = k + (i-1)*E->lmesh.noz+(j-1)*E->lmesh.nox*E->lmesh.noz;
                    if((E->sx[m][1][node]> b->theta_min()) &&
                       (E->sx[m][1][node]< b->theta_max()) &&
                       (E->sx[m][2][node]> b->fi_min()) &&
                       (E->sx[m][2][node]< b->fi_max()) &&
                       (E->sx[m][3][node]> b->ri()) &&
                       (E->sx[m][3][node]< b->ro()))
                    {
                        for(l=0;l<DIM;l++) X[k][n] = E->sx[m][l+1][node];
                        n++;
                    }
                }
#endif
}


void Interior::broadcast(const MPI_Comm& comm, int broadcaster)
{
    // do nothing
}



// version
// $Id: Interior.cc,v 1.1 2003/11/07 01:08:01 tan2 Exp $

// End of file
