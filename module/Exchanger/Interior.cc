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
#include "BoundedBox.h"
#include "Interior.h"


Interior::Interior() :
    BoundedMesh()
{}



Interior::Interior(const BoundedBox& remoteBBox, const All_variables* E) :
    BoundedMesh(E)
{
    std::vector<double> x(DIM);
    int interiornodes,node;
    interiornodes=0;

    for (int m=1;m<=E->sphere.caps_per_proc;m++)
        for(int i=1;i<=E->lmesh.nox;i++) 
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int k=1;k<=E->lmesh.noz;k++)   
                {
                    interiornodes++;
                }

    X_.reserve(interiornodes);
    meshID_.reserve(interiornodes);
    
    for (int m=1;m<=E->sphere.caps_per_proc;m++)
        for(int i=1;i<=E->lmesh.nox;i++) 
	    for(int j=1;j<=E->lmesh.noy;j++)
		for(int k=1;k<=E->lmesh.noz;k++)   
                {
                    node = k + (i-1)*E->lmesh.noz+(j-1)*E->lmesh.nox*E->lmesh.noz;
                    if((E->sx[m][1][node]> remoteBBox[0][0]) &&
                       (E->sx[m][1][node]< remoteBBox[1][0]) &&
                       (E->sx[m][2][node]> remoteBBox[0][1]) &&
                       (E->sx[m][2][node]< remoteBBox[1][1]) &&
                       (E->sx[m][3][node]> remoteBBox[0][2]) &&
                       (E->sx[m][3][node]< remoteBBox[1][2]))
                    {
                        for(int k=0; k<DIM; k++)
                            x[k] = E->sx[m][k+1][node];
                        X_.push_back(x);
                        meshID_.push_back(node);
                    }                    
                }
    X_.shrink();
    X_.print("X");
    meshID_.shrink();
    meshID_.print("meshID");
}


void Interior::broadcast(const MPI_Comm& comm, int broadcaster)
{
    // do nothing
}



// version
// $Id: Interior.cc,v 1.3 2003/11/07 05:47:16 puru Exp $

// End of file
