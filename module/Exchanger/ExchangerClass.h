// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if !defined(pyCitcom_Exchanger_h)
#define pyCitcom_Exchanger_h

#include "mpi.h"


class Boundary;     // declaration only
struct All_variables;

struct Data {
    static const int npass = 8;  // # of arrays to pass
    int size;                    // length of each array
    double *x, *y, *z;    // coordinates
    double *u, *v, *w;    // velocities
    double *T, *P;       // temperature and pressure
};



class Exchanger {

public:
    Exchanger(const MPI_Comm communicator,
	      const MPI_Comm intercomm,
	      const int localLeader,
	      const int remoteLeader,
	      const All_variables *E);
    virtual ~Exchanger();

    void reset_target(const MPI_Comm intercomm,
		      const int receiver);

    virtual void send(int& size);
    virtual void receive(const int size);
    void sendTemperature();
    void receiveTemperature();
    double exchangeTimestep(const double);

    void wait();
    void nowait();

    virtual void gather() = 0;
    virtual void distribute() = 0;
    virtual void interpretate() = 0; // interpolation or extrapolation
    virtual void impose_bc() = 0;    // set bc flag

    virtual void mapBoundary(const Boundary*) = 0;
                                     // create mapping from Boundary object
                                     // to global id array

protected:
    const MPI_Comm comm;
    MPI_Comm intercomm;

    const int localLeader;
    int remoteLeader;

    const All_variables *E;    // CitcomS data structure,
                               // Exchanger only modifies bc flags

    Boundary *boundary;

    Data outgoing;
    Data incoming;

    int rank;

private:
    // disable copy constructor and copy operator
    Exchanger(const Exchanger&);
    Exchanger operator=(const Exchanger&);

};



#endif

// version
// $Id: ExchangerClass.h,v 1.5 2003/09/10 04:03:54 tan2 Exp $

// End of file

