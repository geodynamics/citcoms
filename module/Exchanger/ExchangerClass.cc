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
    E(e)
{}


Exchanger::~Exchanger() {}


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


// version
// $Id: ExchangerClass.cc,v 1.1 2003/09/08 21:47:27 tan2 Exp $

// End of file

