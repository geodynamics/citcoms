// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if !defined(pyCitcom_CoarseGridExchanger_h)
#define pyCitcom_CoarseGridExchanger_h

class Boundary;

#include "ExchangerClass.h"


class CoarseGridExchanger : public Exchanger {

public:
    CoarseGridExchanger(const MPI_Comm communicator,
			const MPI_Comm intercomm,
			const int localrank,
			const int interrank,
			const int localLeader,
			const int remoteLeader,
			const All_variables *E);
    virtual ~CoarseGridExchanger();

  //    virtual void gather();
  //    virtual void distribute();
    virtual void interpretate();
    virtual void impose_bc();
    virtual void mapBoundary();

    void receiveBoundary();
    void interpolate();
    void interpolateTemperature();
};

#endif

// version
// $Id: CoarseGridExchanger.h,v 1.11 2003/09/27 17:12:52 tan2 Exp $

// End of file

