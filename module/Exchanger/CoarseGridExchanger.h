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
    CoarseGridExchanger(const MPI_Comm comm,
			const MPI_Comm intercomm,
			const int leaderRank,
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
// $Id: CoarseGridExchanger.h,v 1.12 2003/09/27 20:30:55 tan2 Exp $

// End of file

