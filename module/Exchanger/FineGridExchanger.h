// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if !defined(pyCitcom_FineGridExchanger_h)
#define pyCitcom_FineGridExchanger_h

class Boundary;

#include "ExchangerClass.h"


class FineGridExchanger : public Exchanger {

public:
    FineGridExchanger(const MPI_Comm communicator,
		      const MPI_Comm intercomm,
		      const int localLeader,
		      const int remoteLeader,
		      const All_variables *E);
    virtual ~FineGridExchanger();

    virtual void gather();
    virtual void distribute();
    virtual void interpretate();
    virtual void impose_bc();
    virtual void mapBoundary(const Boundary*);

    const Boundary* createBoundary();
    int sendBoundary(const Boundary*);

};

#endif

// version
// $Id: FineGridExchanger.h,v 1.5 2003/09/09 20:57:25 tan2 Exp $

// End of file

