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
	      const int localrank,
	      const int interrank,
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
    // Test
    void imposeBC();
    double exchangeTimestep(const double);

    void wait();
    void nowait();

    virtual void gather();
    virtual void distribute();
    virtual void interpretate() = 0; // interpolation or extrapolation
  //    virtual void impose_bc() = 0;    // set bc flag

    virtual void mapBoundary() = 0;
                                     // create mapping from Boundary object
                                     // to global id array

protected:
    const MPI_Comm comm;       // communicator of current solver
    const MPI_Comm intercomm;  // intercommunicator between solvers

    int lrank;                 // proc. rank in comm
    int rank;                  // proc. rank in intercomm

    const int localLeader;     // leader rank (in intercomm) of current solver
    const int remoteLeader;    // leader rank (in intercomm) of another solver

    const All_variables *E;    // CitcomS data structure,
                               // Exchanger only modifies bc flags
    Boundary *boundary;
    

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
// $Id: ExchangerClass.h,v 1.16 2003/09/27 17:12:52 tan2 Exp $

// End of file

