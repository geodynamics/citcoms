// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if !defined(pyCitcom_Exchanger_h)
#define pyCitcom_Exchanger_h

#include "mpi.h"
#include "global_defs.h"

struct Data {
    static const int npass = 8;  // # of arrays to pass
    int size;                    // length of each array
    double *x, *y, *z;    // coordinates
    double *u, *v, *w;    // velocities
    double *T, *P;       // temperature and pressure
};

struct Boundary {
    int bneq;                 // # of boundary nodes
    int *bid2gid;    // bid (local id) -> ID (ie. global id)
    int *bid2proc;   // bid -> proc. rank
};


class Exchanger {

public:
    Exchanger(MPI_Comm communicator,
	      MPI_Comm intercomm,
	      int localLeader,
	      int remoteLeader,
	      const All_variables *E);
    virtual ~Exchanger();

    virtual void send(int& size);
    virtual void receive(const int size);

    virtual void gather() = 0;
    virtual void distribute() = 0;
    virtual void interpretate() = 0; // interpolation or extrapolation
    virtual void impose_bc() = 0;    // set bc flag

protected:
    const MPI_Comm comm;
    const MPI_Comm intercomm;

    const int localLeader;
    const int remoteLeader;

    const All_variables *E;    // CitcomS data structure

    Data outgoing;
    Data incoming;

    Boundary bdry;

private:
    Exchanger(const Exchanger&);
    Exchanger operator=(const Exchanger&);

};

#endif

// version
// $Id: ExchangerClass.h,v 1.1 2003/09/08 21:47:27 tan2 Exp $

// End of file

