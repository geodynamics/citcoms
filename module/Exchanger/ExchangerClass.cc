// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>
#include <fstream>
#include "Boundary.h"
#include "global_defs.h"
#include "ExchangerClass.h"

Exchanger::Exchanger(const MPI_Comm communicator,
		     const MPI_Comm icomm,
		     const int leaderRank,
		     const int local,
		     const int remote,
		     const All_variables *e):
    comm(communicator),
    intercomm(icomm),
    leader(leaderRank),
    localLeader(local),
    remoteLeader(remote),
    E(e),
    boundary(NULL) {

    MPI_Comm_rank(comm, const_cast<int*>(&rank));
    fge_t = cge_t = 0;
}


Exchanger::~Exchanger() {
    std::cout << "in Exchanger::~Exchanger" << std::endl;
}


void Exchanger::reset_target(const MPI_Comm icomm, const int receiver) {
    //intercomm = icomm;
    //remoteLeader = receiver;
}

#if 0
void Exchanger::local_sendVelocities(void) {

    std::cout << "in Exchanger::local_sendVelocities" << std::endl;
    int i,size;

//     int size = outgoing->size;

    loutgoing.size=11;
    size=loutgoing.size;

    for(i=1;i<=size-1;i++) {
	loutgoing.v[0][i]=i*0.01;
	loutgoing.v[1][i]=i*0.01;
	loutgoing.v[2][i]=i*0.01;
    }
    MPI_Send(loutgoing.v[0], size, MPI_DOUBLE, 0, 1, comm);
    MPI_Send(loutgoing.v[1], size, MPI_DOUBLE, 0, 2, comm);
    MPI_Send(loutgoing.v[2], size, MPI_DOUBLE, 0, 3, comm);

    return;
}



void Exchanger::local_receiveVelocities(void) {
    std::cout << "in Exchanger::local_receiveVelocities" << std::endl;

    MPI_Status status;
    int size;
    int worldme,interme;
    int i,nproc;

    // dummy setting
    lincoming.size=5;
    size = lincoming.size;

    MPI_Comm_rank(intercomm,&worldme);
    MPI_Comm_rank(comm,&interme);
    MPI_Comm_size(comm,&nproc);
    std::cout << "interme=" << interme << " worldme=" << worldme << " nproc=" << nproc << std::endl;

    for(i=0;i<size;i++) {
	MPI_Recv(lincoming.v[0], size, MPI_DOUBLE, i, 1, comm, &status);
	/* test */
	std::cout << "interme=" << interme << " worldme=" << worldme
		  << " source=" << i << " Vel_u transferred: size="
		  << size << std::endl;
	MPI_Recv(lincoming.v[1], size, MPI_DOUBLE, i, 2, comm, &status);
	/* test */
	std::cout << "interme=" << interme << " worldme=" << worldme
		  << " source=" << i << " Vel_v transferred: size="
		  << size << std::endl;
	MPI_Recv(lincoming.v[2], size, MPI_DOUBLE, i, 3, comm, &status);
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

//     int size = outgoing->size;

    loutgoing.size=11;
    size=loutgoing.size;

    for(i=1;i<=size-1;i++) {
      loutgoing.T[i]=i*0.01;
    }
    MPI_Send(loutgoing.T, size, MPI_DOUBLE, 0, 1, comm);

    return;
}



void Exchanger::local_receiveTemperature(void) {
    std::cout << "in Exchanger::local_receiveVelocities" << std::endl;

    MPI_Status status;
    int size;
    int worldme,interme;
    int i,nproc;

    // dummy setting
    lincoming.size=5;
    size=lincoming.size;

    MPI_Comm_rank(intercomm,&worldme);
    MPI_Comm_rank(comm,&interme);
    MPI_Comm_size(comm,&nproc);
    std::cout << "interme=" << interme << " worldme=" << worldme << " nproc=" << nproc << std::endl;

    for(i=0;i<size;i++) {
      MPI_Recv(lincoming.T, size, MPI_DOUBLE, i, 1, comm, &status);
      /* test */
      std::cout << "interme=" << interme << " worldme=" << worldme
		<< " source=" << i << " Temp transferred: size="
		<< size << std::endl;
    }

    return;
}
#endif


void Exchanger::createDataArrays() {
    std::cout << "in Exchanger::createDataArrays" << std::endl;

    int size = boundary->size;
    incoming.size = size;
    incoming.T = new double[size];
    outgoing.size = size;
    outgoing.T = new double[size];
    poutgoing.size = size;
    poutgoing.T = new double[size];
    for(int i=0; i < boundary->dim; i++) {
	incoming.v[i] = new double[size];
	outgoing.v[i] = new double[size];
        poutgoing.v[i] = new double[size];
    }
}


void Exchanger::deleteDataArrays() {
    std::cout << "in Exchanger::deleteDataArrays" << std::endl;

      delete [] incoming.T;
      delete [] outgoing.T;
      for(int i=0; i < boundary->dim; i++) {
	  delete [] incoming.v[i];
	  delete [] outgoing.v[i];
      }
}


void Exchanger::sendTemperature(void) {
    std::cout << "in Exchanger::sendTemperature"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader
	      << std::endl;

    if(rank == leader) {

//       std::cout << "nox = " << E->mesh.nox << std::endl;
//       for(int j=0; j < boundary->size; j++)
// 	{
// 	  n=boundary->bid2gid[j];
// 	  outgoing.T[j]=E->T[1][n];

// 	  // Test
// 	  std::cout << "Temperature sent" << std::endl;
// 	  std::cout << j << " " << n << "  " << outgoing.T[j] << std::endl;

// 	}
      MPI_Send(outgoing.T,outgoing.size,MPI_DOUBLE,remoteLeader,0,intercomm);
    }

//     delete outgoing.T;

    return;
}


void Exchanger::receiveTemperature(void) {
    std::cout << "in Exchanger::receiveTemperature"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader
	      << std::endl;
    int n,success;

    MPI_Status status;
    MPI_Request request;

    if(rank == localLeader) {
//       std::cout << "nox = " << E->nox << std::endl;

      MPI_Irecv(incoming.T,incoming.size,MPI_DOUBLE,remoteLeader,0,intercomm,&request);
      std::cout << "Exchanger::receiveTemperature ===> Posted" << std::endl;
    }
    // Test
    MPI_Wait(&request,&status);
    MPI_Test(&request,&success,&status);
    if(success)
      std::cout << "Temperature transfer Succeeded!!" << std::endl;

    for(int j=0; j < boundary->size; j++)
      {
	n=boundary->bid2gid[j];
	std::cout << "Temperature received" << std::endl;
	std::cout << j << " " << n << "  " << incoming.T[j] << std::endl;
      }
    // Don' forget to delete incoming.T
    return;
}


void Exchanger::sendVelocities() {
    std::cout << "in Exchanger::sendVelocities" << std::endl;

    if(rank == leader) {
	int tag = 0;
	for(int i=0; i < boundary->dim; i++) {
	    MPI_Send(outgoing.v[i], outgoing.size, MPI_DOUBLE,
		     remoteLeader, tag, intercomm);
	    tag ++;
	}
    }
}


void Exchanger::receiveVelocities() {
    std::cout << "in Exchanger::receiveVelocities" << std::endl;
    
    if(rank == leader) {
	int tag = 0;
	MPI_Status status;
	for(int i=0; i < boundary->dim; i++) {
            for(int n=0; n < incoming.size; n++)
            {
                if(!((fge_t==0)&&(cge_t==0)))poutgoing.v[i][n]=incoming.v[i][n];
            }
            
	    MPI_Recv(incoming.v[i], incoming.size, MPI_DOUBLE,
		     remoteLeader, tag, intercomm, &status);
	    tag ++;
            if((fge_t==0)&&(cge_t==0))
            {
                for(int n=0; n < incoming.size; n++)poutgoing.v[i][n]=incoming.v[i][n];   
            }
            
	}
    }
    //printDataV(incoming);

    // Don't forget to delete inoming.v
    return;
}


void Exchanger::imposeBC() {

    double N1,N2;
    std::cout << "in Exchanger::imposeBC" << std::endl;

    N1=(cge_t-fge_t)/cge_t;
    N2=fge_t/cge_t;
    
    for(int m=1;m<=E->sphere.caps_per_proc;m++) {
	for(int i=0;i<boundary->size;i++) {
	    int n = boundary->bid2gid[i];
	    int p = boundary->bid2proc[i];
	    if (p == rank) {
		E->sphere.cap[m].VB[1][n] = N1*poutgoing.v[0][i]+N2*incoming.v[0][i];
		E->sphere.cap[m].VB[2][n] = N1*poutgoing.v[1][i]+N2*incoming.v[1][i];
		E->sphere.cap[m].VB[3][n] = N1*poutgoing.v[2][i]+N2*incoming.v[2][i];
		E->node[m][n] = E->node[m][n] | VBX;
		E->node[m][n] = E->node[m][n] | VBY;
		E->node[m][n] = E->node[m][n] | VBZ;
		E->node[m][n] = E->node[m][n] & (~SBX);
		E->node[m][n] = E->node[m][n] & (~SBY);
		E->node[m][n] = E->node[m][n] & (~SBZ);
	    }
	}
    }

    return;
}


void Exchanger::storeTimestep(const double fge_time, const double cge_time) {
    fge_t = fge_time;
    cge_t = cge_time;
}


double Exchanger::exchangeTimestep(const double dt) {
    std::cout << "in Exchanger::exchangeTimestep"
	      << "  rank = " << rank
	      << "  leader = "<< localLeader
	      << "  receiver = "<< remoteLeader << std::endl;
    return exchangeDouble(dt, 1);
}


int Exchanger::exchangeSignal(const int sent) const {
    std::cout << "in Exchanger::exchangeSignal" << std::endl;
    return exchangeInt(sent, 1);
}


// helper functions

double Exchanger::exchangeDouble(const double &sent, const int len) const {
    double received;
    if (rank == leader) {
	const int tag = 350;
	MPI_Status status;

	MPI_Sendrecv((void*)&sent, len, MPI_DOUBLE,
		     remoteLeader, tag,
		     &received, len, MPI_DOUBLE,
		     remoteLeader, tag,
		     intercomm, &status);
    }

    MPI_Bcast(&received, 1, MPI_DOUBLE, leader, comm);
    return received;
}


float Exchanger::exchangeFloat(const float &sent, const int len) const {
    float received;
    if (rank == leader) {
	const int tag = 351;
	MPI_Status status;

	MPI_Sendrecv((void*)&sent, len, MPI_FLOAT,
		     remoteLeader, tag,
		     &received, len, MPI_FLOAT,
		     remoteLeader, tag,
		     intercomm, &status);
    }

    MPI_Bcast(&received, 1, MPI_FLOAT, leader, comm);
    return received;
}


int Exchanger::exchangeInt(const int &sent, const int len) const {
    int received;
    if (rank == leader) {
	const int tag = 352;
	MPI_Status status;

	MPI_Sendrecv((void*)&sent, len, MPI_INT,
		     remoteLeader, tag,
		     &received, len, MPI_INT,
		     remoteLeader, tag,
		     intercomm, &status);
    }

    MPI_Bcast(&received, 1, MPI_INT, leader, comm);
    return received;
}


void Exchanger::printDataT(const Data &data) const {
    for (int n=0; n<data.size; n++) {
	std::cout << "  Data.T:  " << n << ":  "
		  << data.T[n] << std::endl;
    }
}


void Exchanger::printDataV(const Data &data) const {
    for (int n=0; n<data.size; n++) {
	std::cout << "  Data.v:  " << n << ":  ";
	for (int j=0; j<boundary->dim; j++)
	    std::cout << data.v[j][n] << "  ";
	std::cout << std::endl;
    }
}


// version
// $Id: ExchangerClass.cc,v 1.23 2003/09/30 01:45:27 puru Exp $

// End of file

