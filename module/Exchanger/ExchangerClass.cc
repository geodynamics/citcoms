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


void Exchanger::send(int &size) {

    std::cout << "in Exchanger::send" << std::endl;

    /*
    size = outgoing.size;

    MPI_request *request = new MPI_request[outgoing.exchanges-1];
    MPI_Status *status = new MPI_Status[outgoing.exchanges-1];
    int tag = 0;

    MPI_Isend(outgoing.x, size, MPI_DOUBLE, target, tag,
	      intercomm, &request[tag]);
    tag++;


    MPI_Wait(tag, request, status);
    */

    return;
}



void Exchanger::receive(const int size) {
    std::cout << "in Exchanger::receive" << std::endl;

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
// $Id: ExchangerClass.cc,v 1.4 2003/09/10 04:03:54 tan2 Exp $

// End of file

