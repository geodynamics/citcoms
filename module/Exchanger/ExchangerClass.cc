// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>

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

}


Exchanger::~Exchanger() {}


void Exchanger::reset_target(const MPI_Comm icomm, const int receiver) {
    intercomm = icomm;
    remoteLeader = receiver;
}


void Exchanger::sendVelocities(void) {

    std::cout << "in Exchanger::send" << std::endl;
    int i,size;
//     int size = outgoing.size;

    outgoing.size=11;
    size=outgoing.size;

    for(i=1;i<=size-1;i++) {
      outgoing.u[i]=i*0.01;
      outgoing.v[i]=i*0.01;
      outgoing.w[i]=i*0.01;
    }
    MPI_Send(outgoing.u, size, MPI_DOUBLE, 0, 1, intercomm);
    MPI_Send(outgoing.v, size, MPI_DOUBLE, 0, 2, intercomm);
    MPI_Send(outgoing.w, size, MPI_DOUBLE, 0, 3, intercomm);

    return;
}



void Exchanger::receiveVelocities(void) {
    std::cout << "in Exchanger::receive" << std::endl;

    MPI_Status status;
    int size = outgoing.size;
    int i,interme,worldme,nproc;

    MPI_Comm_size(intercomm,&nproc);
    MPI_Comm_rank(intercomm,&interme);
    MPI_Comm_rank(comm,&worldme);

    MPI_Recv(incoming.u, size, MPI_DOUBLE, i, 1, intercomm, &status);
    /* test */
    std::cerr << "interme=" << interme << "worldme=" << world me
	      << "source=" << i << "Vel_u transferred: size=" 
	      << size << "status=" << status << std::endl;
    MPI_Recv(incoming.v, size, MPI_DOUBLE, i, 2, intercomm, &status);
    /* test */
    std::cerr << "interme=" << interme << "worldme=" << world me
	      << "source=" << i << "Vel_v transferred: size=" 
	      << size << "status=" << status << std::endl;
    MPI_Recv(incoming.w, size, MPI_DOUBLE, i, 3, intercomm, &status);
    /* test */
    std::cerr << "interme=" << interme << "worldme=" << world me
	      << "source=" << i << "Vel_w transferred: size=" 
	      << size << "status=" << status << std::endl;
    

    /*
    MPI_request *request = new MPI_request[incoming.exchanges-1];
    MPI_Status *status = new MPI_Status[incoming.exchanges-1];
    int tag = 0;

    MPI_Ireceive(incoming.x, size, MPI_DOUBLE, target, tag,
		 intercomm, &request[tag]);
    tag++;


    int MPI_Wait(tag, request, status);

    */
    return;
}


void Exchanger::sendTemperature() {
    std::cout << "in Exchanger::sendTemperature" << std::endl;
    return;
}


void Exchanger::receiveTemperature() {
    std::cout << "in Exchanger::receiveTemperature" << std::endl;
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
// $Id: ExchangerClass.cc,v 1.5 2003/09/10 18:51:42 ces74 Exp $

// End of file

