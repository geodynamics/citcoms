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

    virtual void gather(const Boundary*);
    virtual void distribute(const Boundary*);
    virtual void interpretate(const Boundary*);
    virtual void impose_bc(const Boundary*);
    virtual void mapBoundary(const Boundary*);

    const Boundary* createBoundary();
    int sendBoundary(const Boundary*);

};

#endif

// version
// $Id: FineGridExchanger.h,v 1.4 2003/09/09 18:25:31 tan2 Exp $

// End of file

