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
    double *x[3];        // coordinates
    double *v[3];        // velocities
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

//     virtual void send(int& size);
//     virtual void receive(const int size);
    void createDataArrays();
    void deleteDataArrays();
    void sendTemperature();
    void receiveTemperature();
    void sendVelocities();
    void receiveVelocities();
    void local_sendVelocities();
    void local_receiveVelocities();
    void local_sendTemperature();
    void local_receiveTemperature();

    double exchangeTimestep(const double);

    void wait();
    void nowait();

    virtual void gather() = 0;
    virtual void distribute() = 0;
    virtual void interpretate() = 0; // interpolation or extrapolation
    virtual void impose_bc() = 0;    // set bc flag

    virtual void mapBoundary() = 0;
                                     // create mapping from Boundary object
                                     // to global id array

protected:
    const MPI_Comm comm;
    const MPI_Comm intercomm;

    const int localLeader;
    const int remoteLeader;

    const All_variables *E;    // CitcomS data structure,
                               // Exchanger only modifies bc flags

    Boundary *boundary;

    int rank;

    struct Data outgoing;
    struct Data incoming;
    struct Data loutgoing;
    struct Data lincoming;


private:
    // disable copy constructor and copy operator
    Exchanger(const Exchanger&);
    Exchanger operator=(const Exchanger&);

};



#endif

// version
// $Id: ExchangerClass.h,v 1.13 2003/09/20 01:32:10 ces74 Exp $

// End of file

