// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>

#include "Boundary.h"
#include "CoarseGridExchanger.h"
#include "global_defs.h"


CoarseGridExchanger::CoarseGridExchanger(MPI_Comm communicator,
					 MPI_Comm icomm,
					 int local,
					 int remote,
					 const All_variables *e):
    Exchanger(communicator, icomm, local, remote, e)
{
    std::cout << "in CoarseGridExchanger::CoarseGridExchanger" << std::endl;
}



CoarseGridExchanger::~CoarseGridExchanger() {
    std::cout << "in CoarseGridExchanger::~CoarseGridExchanger" << std::endl;
}



void CoarseGridExchanger::gather() {
    std::cout << "in CoarseGridExchanger::gather" << std::endl;

    int me,nproc;

    MPI_Comm_rank(comm,&me);
    MPI_Comm_size(comm,&nproc);

    std::cout << "me= " << me << " nproc=" << nproc << std::endl;

    if(nproc>1) {
      if(me>0)
	local_sendVelocities();
      if(me==0)
	local_receiveVelocities();
      MPI_Barrier(intercomm);
    }
    else
      std::cout << "Don't need to run gather since nproc is " << nproc << std::endl;

    return;
}



void CoarseGridExchanger::distribute() {
    std::cout << "in CoarseGridExchanger::distribute" << std::endl;

    int me,nproc;

    MPI_Comm_rank(comm,&me);
    MPI_Comm_size(comm,&nproc);

    std::cout << "me= " << me << " nproc=" << nproc << std::endl;

    if(nproc>1) {
      if(me>0)
	local_sendVelocities();
      if(me==0)
	local_receiveVelocities();
      MPI_Barrier(intercomm);
    }
    else {
      std::cout << "in CoarseGridExchanger::distribute" << std::endl;
      std::cout << "Don't need to run gather since nproc is " << nproc << std::endl;
    }
    return;
}



void CoarseGridExchanger::interpretate() {
    std::cout << "in CoarseGridExchanger::interpretate" << std::endl;
}

void CoarseGridExchanger::interpolate() {
  
  double finex[3],crsex[24],xi[3],shape[8];
  int n,node;
  
//   outgoing.size=boundary->size;
//   outgoing.T = new double[outgoing.size];
//     for(int i=0; i < boundary->dim; i++)
//       {
// 	outgoing.v[i] = new double[outgoing.size];
//       }

    std::cout << "in CoarseGridExchanger::interpolate" << std::endl;
//     for(int i=0;i<boundary->size;i++)
//       {
// 	std::cout << "i = " << i << " boundary->bid2crseelem = " << boundary->bid2crseelem[i] << std::endl;	
//       }

    for(int i=0;i<boundary->size;i++)
      {
	n=boundary->bid2crseelem[i];
	for(int j=0; j < boundary->dim ;j++)
	  {
// 	    std::cout << "i = " << i << " j = " << j << " boundary->X = " << boundary->X[j][i] << std::endl;	
	    finex[j]=boundary->X[j][i];
	    for(int k=0; k< 8 ;k++)
	      {
		node=E->IEN[E->mesh.levmax][1][n].node[k+1];
		crsex[k*3+j]=E->X[E->mesh.levmax][1][j+1][node];
	      }
	  }
	
	xi[0]=(crsex[3]-finex[0])/(crsex[3]-crsex[0]);
	xi[1]=(crsex[10]-finex[1])/(crsex[10]-crsex[1]);
	xi[2]=(crsex[14]-finex[2])/(crsex[14]-crsex[2]);
// 	std::cout << "xi[0] = " << xi[0] << " "	
// 		  << "xi[1] = " << xi[1] << " "	
// 		  << "xi[2] = " << xi[2] << " "	
// 		  << std::endl;
	shape[0]=(1.-xi[0])*(1.-xi[1])*(1.-xi[2]);
	shape[1]=xi[0]*(1.-xi[1])*(1.-xi[2]);
	shape[2]=xi[0]*xi[1]*(1.-xi[2]);
	shape[3]=(1.-xi[0])*xi[1]*(1.-xi[2]);

	shape[4]=(1.-xi[0])*(1.-xi[1])*xi[2];
	shape[5]=xi[0]*(1.-xi[1])*xi[2];
	shape[6]=xi[0]*xi[1]*xi[2];
	shape[7]=(1.-xi[0])*xi[1]*xi[2];
	
	outgoing.T[i]=outgoing.v[0][i]=outgoing.v[1][i]=outgoing.v[2][i];

	for(int k=0; k< 8 ;k++)
	  {
	    node=E->IEN[E->mesh.levmax][1][n].node[k+1];
// 	    std::cout << "node = " << node << " "
// 		      << "k = " << k << " "
// 		      << "shape = " << shape[k] << " "
// 		      << "T = " << E->T[E->sphere.caps_per_proc][node]<<" "
// 		      << "v1 = " << E->V[E->sphere.caps_per_proc][1][node]<<" "
// 		      << "v2 = " << E->V[E->sphere.caps_per_proc][2][node]<<" "
// 		      << "v3 = " << E->V[E->sphere.caps_per_proc][3][node]<<" "
// 		      << std::endl;
	    outgoing.T[i]+=shape[k]*E->T[E->sphere.caps_per_proc][node];
	    outgoing.v[0][i]+=shape[k]*E->V[E->sphere.caps_per_proc][1][node];
	    outgoing.v[1][i]+=shape[k]*E->V[E->sphere.caps_per_proc][2][node];
	    outgoing.v[2][i]+=shape[k]*E->V[E->sphere.caps_per_proc][3][node];
	  }
      }

    // Test
//     std::cout << "in CoarseGridExchanger::interpolated fields" << std::endl;
//     for(int i=0;i<boundary->size;i++)
//       {
// 	std::cout << i << " " << outgoing.T[i] << std::endl;
//       }

    return;
}


void CoarseGridExchanger::impose_bc() {
    std::cout << "in CoarseGridExchanger::impose_bc" << std::endl;

}


void CoarseGridExchanger::receiveBoundary() {
    std::cout << "in CoarseGridExchanger::receiveBoundary"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader << std::endl;

    if (rank == localLeader) {
	int tag = 0;
	MPI_Status status;
	int size;

 	MPI_Recv(&size, 1, MPI_INT,
 		 remoteLeader, tag, intercomm, &status);
	tag ++;

	boundary = new Boundary(size);

  	MPI_Recv(boundary->connectivity, size, MPI_INT,
  		 remoteLeader, tag, intercomm, &status);
  	//boundary->printConnectivity();
 	tag ++;

	for (int i=0; i<boundary->dim; i++) {
  	    MPI_Recv(boundary->X[i], size, MPI_DOUBLE,
  		     remoteLeader, tag, intercomm, &status);
	    tag ++;
	}
 	MPI_Recv(&boundary->theta_max, 1, MPI_DOUBLE,
 		 remoteLeader, tag, intercomm, &status);
 	tag ++;
 	MPI_Recv(&boundary->theta_min, 1, MPI_DOUBLE,
 		 remoteLeader, tag, intercomm, &status);
 	tag ++;
 	MPI_Recv(&boundary->fi_max, 1, MPI_DOUBLE,
 		 remoteLeader, tag, intercomm, &status);
 	tag ++;
 	MPI_Recv(&boundary->fi_min, 1, MPI_DOUBLE,
 		 remoteLeader, tag, intercomm, &status);
 	tag ++;
 	MPI_Recv(&boundary->ro, 1, MPI_DOUBLE,
 		 remoteLeader, tag, intercomm, &status);
 	tag ++;
 	MPI_Recv(&boundary->ri, 1, MPI_DOUBLE,
 		 remoteLeader, tag, intercomm, &status);
 	tag ++;

	// test 
	std::cout << "in CoarseGridExchanger::receiveBoundary" << std::endl;
	std::cout << "Grid Bounds transferred to Coarse Grid" << std::endl;
	std::cout << "theta= " << boundary->theta_min<< "   " << boundary->theta_max << std::endl;
	std::cout << "fi   = " << boundary->fi_min << "   " << boundary->fi_max << std::endl;
	std::cout << "r    = " << boundary->ri << "   " << boundary->ro  << std::endl;

	
	//boundary->printX();
    }

}

void CoarseGridExchanger::getBid2crseelem() {
    std::cout << "in FineGridExchanger::getBid2crseelem" << std::endl;

    boundary->getBid2crseelem(E);

    return;
}


void CoarseGridExchanger::mapBoundary() {
    std::cout << "in CoarseGridExchanger::mapBoundary" << std::endl;
    boundary->mapCoarseGrid(E, localLeader);
}



// version
// $Id: CoarseGridExchanger.cc,v 1.13 2003/09/21 22:24:00 ces74 Exp $

// End of file
