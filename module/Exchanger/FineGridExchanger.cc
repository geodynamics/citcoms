// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <iostream>

#include "FineGridExchanger.h"



FineGridExchanger::FineGridExchanger(MPI_Comm communicator,
				     MPI_Comm icomm,
				     int local,
				     int remote,
				     const All_variables *e)
     :
    Exchanger(communicator, icomm, local, remote, e)
    /*
    comm(communicator),
    intercomm(icomm),
    localLeader(local),
    remoteLeader(remote),
    E(e)
    */
{
    std::cout << "in FineGridExchanger::FineGridExchanger" << std::endl;
    /*
    comm = communicator;
    intercomm = icomm;
    localLeader = local;
    remoteLeader = remote;
    E = e;
    */
    return;
}



FineGridExchanger::~FineGridExchanger() {
    std::cout << "in FineGridExchanger::~FineGridExchanger" << std::endl;
}



void FineGridExchanger::set_target(int, int, int) {
    std::cout << "in FineGridExchanger::setup" << std::endl;
}


void FineGridExchanger::gather() {
    std::cout << "in FineGridExchanger::gather" << std::endl;
}



void FineGridExchanger::distribute() {
    std::cout << "in FineGridExchanger::distribute" << std::endl;
}



void FineGridExchanger::interpretate() {
    std::cout << "in FineGridExchanger::interpretate" << std::endl;
}




void FineGridExchanger::impose_bc() {
    std::cout << "in FineGridExchanger::impose_bc" << std::endl;

}



// version
// $Id: FineGridExchanger.cc,v 1.2 2003/09/08 21:47:27 tan2 Exp $

// End of file
