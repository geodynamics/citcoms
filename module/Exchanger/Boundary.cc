// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <iostream>

#include "global_defs.h"
#include "Boundary.h"
#include "ExchangerClass.h"

Boundary::Boundary(const int n):
    size(n),
    connectivity_(new int[size]) {

    // use auto_ptr for exception-proof
    for(int i=0; i<dim; i++)
	X_[i] = std::auto_ptr<double>(new double[size]);

    // setup traditional pointer for convenience
    connectivity = connectivity_.get();
    for(int i=0; i<dim; i++)
	X[i] = X_[i].get();

  bid2proc= new int[size];
  bid2gid= new int[size];
}



Boundary::~Boundary() {};


void Boundary::init(const All_variables *E) {
  int nodes,node1,node2;
  nodes=0;
    // test
    for(int j=0; j<size; j++)
	connectivity[j] = j;

    //  for two YOZ planes 

    if (E->parallel.me_loc[1]==0 || E->parallel.me_loc[1]==E->parallel.nprocx-1)
      for (int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int j=1;j<=E->lmesh.noy-1;j++)
	  for(int i=1;i<=E->lmesh.noz-1;i++)  {
	    node1 = i + (j-1)*E->lmesh.noz*E->lmesh.nox;
	    node2 = node1 + (E->lmesh.nox-1)*E->lmesh.noz;
	    
	    if (E->parallel.me_loc[1]==0 )  {
	      for(int k=0;k<3;k++)X[k][nodes]=E->X[E->mesh.levmax][1][k+1][node1];
	      nodes++;	      
	    }
	    if (E->parallel.me_loc[1]==E->parallel.nprocx-1)  {
	      for(int k=0;k<3;k++)X[k][nodes]=E->X[E->mesh.levmax][1][k+1][node2];
	      nodes++;
	    }
	  }

    //  for two XOZ planes

    if (E->parallel.me_loc[2]==0 || E->parallel.me_loc[2]==E->parallel.nprocy-1)
      for (int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int j=1;j<=E->lmesh.nox-1;j++)
	  for(int i=1;i<=E->lmesh.noz-1;i++)  {
	    node1 = i + (j-1)*E->lmesh.noz;
	    node2 = node1 + (E->lmesh.noy-1)*E->lmesh.noz*E->lmesh.nox;	    
	    if (E->parallel.me_loc[2]==0 )  {
	      for(int k=0;k<3;k++)X[k][nodes]=E->X[E->mesh.levmax][1][k+1][node1];
	      nodes++;	      
	    }
	    if (E->parallel.me_loc[2]==E->parallel.nprocy-1)  {
	      for(int k=0;k<3;k++)X[k][nodes]=E->X[E->mesh.levmax][1][k+1][node2];
	      nodes++;
	    }
	  }
    //  for two XOY planes

    if (E->parallel.me_loc[3]==0 || E->parallel.me_loc[3]==E->parallel.nprocz-1)
      for (int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int j=1;j<=E->lmesh.noy-1;j++)
	  for(int i=1;i<=E->lmesh.nox-1;i++)  {
	    node1 = 1 + (i-1)*E->lmesh.noz+(j-1)*E->lmesh.nox*E->lmesh.noz;
	    node2 = node1 + E->lmesh.noz-1;
	    
	    if (E->parallel.me_loc[3]==0 )  {
	      for(int k=0;k<3;k++)X[k][nodes]=E->X[E->mesh.levmax][1][k+1][node1];
	      nodes++;	      
	    }
	    if (E->parallel.me_loc[3]==E->parallel.nprocz-1)  {
	      for(int k=0;k<3;k++)X[k][nodes]=E->X[E->mesh.levmax][1][k+1][node2];
	      nodes++;
	    }
	  }
    if(nodes != size) std::cout << " nodes != size "; 

    //    for(int i=0; i<dim; i++)
    //	for(int j=0; j<size; j++) {
    //	    X[i][j] = i+j;
    //	}
}



void Boundary::map(const All_variables *E, int localLeader) {
  
  for(int i=0; i<size; i++)
    bid2proc[i]=localLeader;

  int n=0;
  for(int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int k=1;k<=E->lmesh.noy;k++)
	    for(int j=1;j<=E->lmesh.nox;j++)
		for(int i=1;i<=E->lmesh.noz;i++)  {
		    int node = i + (j-1)*E->lmesh.noz
			     + (k-1)*E->lmesh.noz*E->lmesh.nox;
		    if((k==1)||(k==E->lmesh.noy)||(j==1)||(j==E->lmesh.nox)||(i==1)||(i==E->lmesh.noz))
		    {
		      bid2gid[n]=node;
		      n++;
		    }
		}
}



void Boundary::printConnectivity() const {
    for(int i=0; i<dim; i++) {
	std::cout << "dimension: " << i << std::endl;
	int *c = connectivity;
	for(int j=0; j<size; j++)
	    std::cout << "    " << j << ":  " << c[j] << std::endl;
    }
}


// version
// $Id: Boundary.cc,v 1.4 2003/09/10 23:40:09 puru Exp $

// End of file
