// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>

#include "mpi.h"
#include "global_defs.h"

class Exchanger {

public:
    Exchanger(const All_variables *E);
    ~Exchanger();

    void set_target(const MPI_Comm intercomm, const int receiver_rank);
    void send(int size);
    void receive(const int size);

    virtual void gather() = 0;
    virtual void distribute() = 0;
    virtual void interpretate() = 0; // interpolation or extrapolation
    virtual void impose_bc() = 0;

protected:
    MPI_Comm comm;
    int master = 0;

    MPI_Comm intercomm;
    int target = 0;

    Data outgoing;
    Data incoming;

    int bneq;           // # of boundary equation
    int *bid2gid;    // bid (local id) -> ID (ie. global id)
    int *bid2proc;   // bid -> proc. rank

};



void Exchanger::send(int &size) {

    cout << "in Exchanger::send" << std::endl;

    size = outgoing.size;

    MPI_request *request = new MPI_request[outgoing.exchanges-1];
    MPI_Status *status = new MPI_Status[outgoing.exchanges-1];
    int tag = 0;

    MPI_Isend(outgoing.x, size, MPI_DOUBLE, target, tag,
	      intercomm, &request[tag]);
    tag++;


    MPI_Wait(tag, request, status);

    return;
}



void Exchanger::receive(const int size) {
    cout << "in Exchanger::receive" << std::endl;

    MPI_request *request = new MPI_request[incoming.exchanges-1];
    MPI_Status *status = new MPI_Status[incoming.exchanges-1];
    int tag = 0;

    MPI_Ireceive(incoming.x, size, MPI_DOUBLE, target, tag,
		 intercomm, &request[tag]);
    tag++;


    int MPI_Wait(tag, request, status);

    return;
}




// version
// $Id: Exchanger.cc,v 1.1 2003/09/06 23:44:22 tan2 Exp $

// End of file

