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
                       appendX(E, m, node);
                    }                    
                }
    
}


void Interior::broadcast(const MPI_Comm& comm, int broadcaster)
{
    // do nothing
}



// version
// $Id: Interior.cc,v 1.2 2003/11/07 05:14:16 puru Exp $

// End of file
