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

    virtual void gather(const Boundary*);
    virtual void distribute(const Boundary*);
    virtual void interpretate(const Boundary*);
    virtual void impose_bc(const Boundary*);
    virtual void mapBoundary(const Boundary*);

    const Boundary* receiveBoundary();

};

#endif

// version
// $Id: CoarseGridExchanger.h,v 1.3 2003/09/09 18:25:31 tan2 Exp $

// End of file

