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
    valarray<double> x, y, z;    // coordinates
    valarray<double> u, v, w;    // velocities
    valarray<double> T, P;       // temperature and pressure
};

struct Boundary {
    int bneq;                 // # of boundary nodes
    valarray<int> bid2gid;    // bid (local id) -> ID (ie. global id)
    valarray<int> bid2proc;   // bid -> proc. rank
}


class Exchanger {

public:
    Exchanger(const All_variables *E);
    virtual ~Exchanger();

    virtual void set_target(const MPI_Comm comm,
			    const MPI_Comm intercomm,
			    const int receiver);
    virtual void send(int& size);
    virtual void receive(const int size);

    virtual void gather() = 0;
    virtual void distribute() = 0;
    virtual void interpretate() = 0; // interpolation or extrapolation
    virtual void impose_bc() = 0;    // set bc flag

protected:
    const All_variables *E;    // CitcomS data structure

    MPI_Comm comm;
    int localLeader = 0;

    MPI_Comm intercomm;
    int remoteLeader = 0;

    Data outgoing;
    Data incoming;

    Boundary bdry;

private:
    Exchanger(const Exchanger);


};

#endif

// version
// $Id: Exchanger.h,v 1.1 2003/09/06 23:44:22 tan2 Exp $

// End of file

