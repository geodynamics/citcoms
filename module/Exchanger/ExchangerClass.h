// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_Exchanger_h)
#define pyCitcom_Exchanger_h

#include <memory>
#include "mpi.h"

template <int dim> class Array2D;
class Boundary;
struct All_variables;

typedef std::auto_ptr<Array2D<1> > Temper;
typedef std::auto_ptr<Array2D<3> > Velo;

class Exchanger {

public:
    Exchanger(const MPI_Comm comm,
	      const MPI_Comm intercomm,
	      const int leader,
	      const int localLeader,
	      const int remoteLeader,
	      const All_variables *E);
    virtual ~Exchanger();

    void createDataArrays();
    void deleteDataArrays();

    void initTemperature();
    void sendTemperature();
    void receiveTemperature();
    void sendVelocities();
    void receiveVelocities();

    void imposeConstraint();
    void imposeBC();
    void setBCFlag();

    void storeTimestep(const double fge_time, const double cge_time);
    double exchangeTimestep(const double) const;
    int exchangeSignal(const int) const;

    virtual void gather() = 0;
    virtual void distribute() = 0;
    virtual void interpretate() = 0;  // interpolate or extrapolate
    //virtual void imposeBC() = 0;      // set bc flag
    virtual void mapBoundary() = 0;   // create mapping from Boundary object
                                      // to global id array

protected:
    const MPI_Comm comm;       // communicator of current solver
    const MPI_Comm intercomm;  // intercommunicator between solvers

    const int rank;            // proc. rank in comm
    const int leader;          // leader rank (in comm) of current solver

    const int localLeader;     // leader rank (in intercomm) of current solver
    const int remoteLeader;    // leader rank (in intercomm) of another solver

    const All_variables *E;    // CitcomS data structure,
                               // Exchanger only modifies bc flags directly
	                       // and id array indirectly
    Boundary *boundary;

    Temper outgoingT;
    Temper incomingT;

    Velo localV;
    Velo outgoingV;
    Velo incomingV;
    Velo old_incomingV;

    double fge_t, cge_t;

private:
    double exchangeDouble(const double &sent, const int len) const;
    float exchangeFloat(const float &sent, const int len) const;
    int exchangeInt(const int &sent, const int len) const;

    void computeWeightedNormal(double* nwght) const;
    double computeOutflow(const Velo& V, const double* nwght) const;
    void reduceOutflow(const double outflow, const double* nwght);

    // disable copy constructor and copy operator
    Exchanger(const Exchanger&);
    Exchanger& operator=(const Exchanger&);


};



#endif

// version
// $Id: ExchangerClass.h,v 1.27 2003/10/10 18:14:49 tan2 Exp $

// End of file

