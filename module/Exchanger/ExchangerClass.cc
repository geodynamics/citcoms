// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>

#include "Boundary.h"
#include "global_defs.h"
#include "ExchangerClass.h"

Exchanger::Exchanger(MPI_Comm communicator,
		     MPI_Comm icomm,
		     int local,
		     int remote,
		     const All_variables *e):
    comm(communicator),
    intercomm(icomm),
    localLeader(local),
    remoteLeader(remote),
    E(e),
    boundary(NULL) {

    MPI_Comm_rank(intercomm, &rank);

//     int size;
//     MPI_Comm_size(intercomm, &size);
//     std::cout << "intercomm: rank = " << rank
// 	      << "  size = " << size << std::endl;

//     int localrank;
//     MPI_Comm_rank(comm, &localrank);
//     MPI_Comm_size(intercomm, &size);
//     std::cout << "localcomm: rank = " << localrank
// 	      << "  size = " << size << std::endl;


}


Exchanger::~Exchanger() {
    std::cout << "in Exchanger::~Exchanger" << std::endl;
}


void Exchanger::reset_target(const MPI_Comm icomm, const int receiver) {
    //intercomm = icomm;
    //remoteLeader = receiver;
}


void Exchanger::local_sendVelocities(void) {

    std::cout << "in Exchanger::local_sendVelocities" << std::endl;
    int i,size;
    struct Data *loutgoing;

//     int size = outgoing->size;

    loutgoing->size=11;
    size=loutgoing->size;

    for(i=1;i<=size-1;i++) {
      loutgoing->v[0][i]=i*0.01;
      loutgoing->v[1][i]=i*0.01;
      loutgoing->v[2][i]=i*0.01;
    }
    MPI_Send(loutgoing->v[0], size, MPI_DOUBLE, 0, 1, comm);
    MPI_Send(loutgoing->v[1], size, MPI_DOUBLE, 0, 2, comm);
    MPI_Send(loutgoing->v[2], size, MPI_DOUBLE, 0, 3, comm);

    return;
}



void Exchanger::local_receiveVelocities(void) {
    std::cout << "in Exchanger::local_receiveVelocities" << std::endl;

    MPI_Status status;
    int size;
    int worldme,interme;
    int i,nproc;
    struct Data *lincoming;

    // dummy setting
    lincoming->size=5;
    size = lincoming->size;

    MPI_Comm_rank(intercomm,&worldme);
    MPI_Comm_rank(comm,&interme);
    MPI_Comm_size(comm,&nproc);
    std::cout << "interme=" << interme << " worldme=" << worldme << " nproc=" << nproc << std::endl;
      
    for(i=0;i<size;i++) {
      MPI_Recv(lincoming->v[0], size, MPI_DOUBLE, i, 1, comm, &status);
      /* test */
      std::cout << "interme=" << interme << " worldme=" << worldme
		<< " source=" << i << " Vel_u transferred: size="
		<< size << std::endl;
      MPI_Recv(lincoming->v[1], size, MPI_DOUBLE, i, 2, comm, &status);
      /* test */
      std::cout << "interme=" << interme << " worldme=" << worldme
		<< " source=" << i << " Vel_v transferred: size="
		<< size << std::endl;
      MPI_Recv(lincoming->v[2], size, MPI_DOUBLE, i, 3, comm, &status);
      /* test */
      std::cout << " interme=" << interme << " worldme=" << worldme
		<< " source=" << i << " Vel_w transferred: size="
		<< size << std::endl;
    }

    /*
    MPI_request *request = new MPI_request[incoming->exchanges-1];
    MPI_Status *status = new MPI_Status[incoming->exchanges-1];
    int tag = 0;

    MPI_Ireceive(incoming->x, size, MPI_DOUBLE, target, tag,
		 intercomm, &request[tag]);
    tag++;


    int MPI_Wait(tag, request, status);

    */
    return;
}


void Exchanger::local_sendTemperature(void) {

    std::cout << "in Exchanger::sendTemperature" << std::endl;
    int i,size;
    struct Data *loutgoing;

//     int size = outgoing->size;

    loutgoing->size=11;
    size=loutgoing->size;

    for(i=1;i<=size-1;i++) {
      loutgoing->T[i]=i*0.01;
    }
    MPI_Send(loutgoing->T, size, MPI_DOUBLE, 0, 1, comm);

    return;
}



void Exchanger::local_receiveTemperature(void) {
    std::cout << "in Exchanger::local_receiveVelocities" << std::endl;

    MPI_Status status;
    int size;
    int worldme,interme;
    int i,nproc;
    struct Data *lincoming;

    // dummy setting
    lincoming->size=5;
    size=lincoming->size;

    MPI_Comm_rank(intercomm,&worldme);
    MPI_Comm_rank(comm,&interme);
    MPI_Comm_size(comm,&nproc);
    std::cout << "interme=" << interme << " worldme=" << worldme << " nproc=" << nproc << std::endl;
      
    for(i=0;i<size;i++) {
      MPI_Recv(lincoming->T, size, MPI_DOUBLE, i, 1, comm, &status);
      /* test */
      std::cout << "interme=" << interme << " worldme=" << worldme
		<< " source=" << i << " Temp transferred: size="
		<< size << std::endl;
    }

    return;
}

void Exchanger::createDataArrays(void) {
    std::cout << "in Exchanger::createDataArrays"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader
	      << std::endl;

    if(rank == localLeader) {
    }

    return;
}

void Exchanger::deleteDataArrays(void) {
    std::cout << "in Exchanger::deleteDataArrays"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader
	      << std::endl;

    if(rank == localLeader) {
    }

    return;
}

void Exchanger::sendTemperature(void) {
    std::cout << "in Exchanger::sendTemperature" 
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader
	      << std::endl;
    int n;
    struct Data outgoing;

    std::cout << "boundary->size = " << boundary->size << std::endl;
    outgoing.size=boundary->size;
    std::cout << "outgoing->size = " << outgoing.size << std::endl;
    outgoing.T = new double[outgoing.size];
    
    if(rank == localLeader) {
      
//       std::cout << "nox = " << E->mesh.nox << std::endl;
      for(int j=0; j < boundary->size; j++)
	{
	  n=boundary->bid2gid[j];
	  outgoing.T[j]=E->T[1][n];

	  // Test
	  std::cout << "Sending Temperature" << std::endl;
	  std::cout << j << " " << n << "  " << outgoing.T[j] << std::endl;
	    
	}
      MPI_Send(outgoing.T,outgoing.size,MPI_DOUBLE,remoteLeader,0,intercomm);
    }

    delete outgoing.T;

    return;
}


void Exchanger::receiveTemperature(void) {
    std::cout << "in Exchanger::receiveTemperature" 
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader
	      << std::endl;
    int n,success;
    struct Data *incoming;

    incoming->size=boundary->size;
    incoming->T = new double[incoming->size];

    MPI_Status status;
    MPI_Request request;
    
    if(rank == localLeader) {
//       std::cout << "nox = " << E->nox << std::endl;

      MPI_Irecv(incoming->T,incoming->size,MPI_DOUBLE,remoteLeader,0,intercomm,&request);
      std::cout << "Exchanger::receiveTemperature" << std::endl;
      std::cout << "Receive posted " << std::endl;
    }
    // Test
    MPI_Wait(&request,&status);
    MPI_Test(&request,&success,&status);
    if(success)
      std::cout << "Sending Temperature Succeeded!!" << std::endl;

    for(int j=0; j < boundary->size; j++)
      {
	n=boundary->bid2gid[j];
	std::cout << "Received Temperature" << std::endl;
	std::cout << j << " " << n << "  " << incoming->T[j] << std::endl;
      }
    // Don' forget to delete incoming->T
    return;
}

void Exchanger::sendVelocities() {
    std::cout << "in Exchanger::sendVelocities" 
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader
	      << std::endl;

    int n;
    struct Data *outgoing;

    outgoing->size=boundary->size;
    for(int i=0; i < boundary->dim; i++)
      {
	outgoing->v[i] = new double[outgoing->size];
      }
    for(int j=0; j < boundary->size; j++)
	{
	  n=boundary->bid2gid[j];
	  for(int i=0; i< boundary->dim; i++)
	    {
	      outgoing->v[i][j]=E->V[E->sphere.caps_per_proc][i][n];	 
	    }
	}
    if(rank == localLeader) {
      MPI_Send(outgoing->v[0],outgoing->size,MPI_DOUBLE,remoteLeader,1,intercomm);
      MPI_Send(outgoing->v[1],outgoing->size,MPI_DOUBLE,remoteLeader,2,intercomm);
      MPI_Send(outgoing->v[2],outgoing->size,MPI_DOUBLE,remoteLeader,3,intercomm);
    }
    for(int i=0; i < boundary->dim; i++)
      {
	delete outgoing->v[i];
      }
    return;
}


void Exchanger::receiveVelocities() {
    std::cout << "in Exchanger::receiveVelocities" 
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader
	      << std::endl;

    struct Data *incoming;

    incoming->size=boundary->size;
    MPI_Status status;
    for(int i=0; i < boundary->dim; i++)
      {
	incoming->v[i] = new double[incoming->size];
      }
    if(rank == localLeader) {
      MPI_Recv(incoming->v[0],incoming->size,MPI_DOUBLE,remoteLeader,1,intercomm,&status);
      MPI_Recv(incoming->v[1],incoming->size,MPI_DOUBLE,remoteLeader,2,intercomm,&status);
      MPI_Recv(incoming->v[2],incoming->size,MPI_DOUBLE,remoteLeader,3,intercomm,&status);
    }
    // Don't forget to delete inoming.v
    return;
}


double Exchanger::exchangeTimestep(const double dt) {
    std::cout << "in Exchanger::exchangeTimestep"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader << std::endl;
    double remotedt = dt;

    if (rank == localLeader) {
	const int tag = 0;
	MPI_Status status;

	MPI_Sendrecv_replace(&remotedt, 1, MPI_DOUBLE,
			     remoteLeader, tag,
			     remoteLeader, tag,
			     intercomm, &status);
    }
    return remotedt;
}


static const int WAIT_TAG = 356;

void Exchanger::wait() {
    std::cout << "in Exchanger::wait" << std::endl;
    if (rank == localLeader) {
	int junk;
	MPI_Status status;

	MPI_Recv(&junk, 1, MPI_INT,
		 remoteLeader, WAIT_TAG, intercomm, &status);
    }
    MPI_Barrier(comm);  // wait until leader has received signal
    return;
}


void Exchanger::nowait() {
    std::cout << "in Exchanger::nowait" << std::endl;
    if (rank == localLeader) {
	int junk = 0;

	MPI_Send(&junk, 1, MPI_INT,
		 remoteLeader, WAIT_TAG, intercomm);
    }
    return;
}


// version
// $Id: ExchangerClass.cc,v 1.9 2003/09/19 06:32:42 ces74 Exp $

// End of file

