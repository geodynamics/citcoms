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

struct All_variables;
template <int dim> class Array2D;
class Boundary;
class Mapping;


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

    void storeTimestep(const double fge_time, const double cge_time);
    double exchangeTimestep(const double) const;
    int exchangeSignal(const int) const;

    virtual void gather() = 0;
    virtual void distribute() = 0;
    virtual void interpretate() = 0;  // interpolate or extrapolate
    virtual void mapBoundary() = 0;   // create mapping from Boundary object
                                      // to global id array
    virtual void createMapping() = 0;

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

    Mapping *mapping;

    typedef std::auto_ptr<Array2D<1> > Temper;
    typedef std::auto_ptr<Array2D<3> > Velo;

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

    // disable copy constructor and copy operator
    Exchanger(const Exchanger&);
    Exchanger& operator=(const Exchanger&);


};



#endif

// version
// $Id: ExchangerClass.h,v 1.28 2003/10/11 00:38:46 tan2 Exp $

// End of file

