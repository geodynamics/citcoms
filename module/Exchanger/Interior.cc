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
    BoundedMesh()
{
    bbox_ = remoteBBox;
    
    std::vector<double> x(DIM);
    int node;
    
    X_.reserve(E->lmesh.nno);
    meshID_.reserve(E->lmesh.nno);
    
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

// version
// $Id: Interior.cc,v 1.4 2003/11/07 20:41:32 puru Exp $

// End of file
