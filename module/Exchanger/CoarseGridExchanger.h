// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_CoarseGridExchanger_h)
#define pyCitcom_CoarseGridExchanger_h

#include "ExchangerClass.h"

class CoarseGridMapping;


class CoarseGridExchanger : public Exchanger {
    CoarseGridMapping* cgmapping;

public:
    CoarseGridExchanger(const MPI_Comm comm,
			const MPI_Comm intercomm,
			const int leader,
			const int localLeader,
			const int remoteLeader,
			const All_variables *E);
    virtual ~CoarseGridExchanger();

    virtual void gather();
    virtual void distribute();
    virtual void interpretate();
    virtual void mapBoundary();
    virtual void createMapping();

    void receiveBoundary();
    void interpolateTemperature();
};

#endif

// version
// $Id: CoarseGridExchanger.h,v 1.14 2003/10/11 00:38:46 tan2 Exp $

// End of file

