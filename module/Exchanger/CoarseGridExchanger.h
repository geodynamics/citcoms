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
			const int localLeader,
			const int remoteLeader,
			const All_variables *E);
    virtual ~CoarseGridExchanger();

    virtual void gather();
    virtual void distribute();
    virtual void interpretate();
    virtual void impose_bc();
    virtual void mapBoundary(Boundary*);

    const Boundary* receiveBoundary();

};

#endif

// version
// $Id: CoarseGridExchanger.h,v 1.5 2003/09/10 21:11:09 puru Exp $

// End of file

