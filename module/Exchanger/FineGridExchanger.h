// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_FineGridExchanger_h)
#define pyCitcom_FineGridExchanger_h

#include "ExchangerClass.h"

class AreaWeightedNormal;
class FineGridMapping;


class FineGridExchanger : public Exchanger {
    FineGridMapping* fgmapping;
    AreaWeightedNormal* awnormal;

public:
    FineGridExchanger(const MPI_Comm comm,
		      const MPI_Comm intercomm,
		      const int leader,
		      const int remoteLeader,
		      const All_variables *E);
    virtual ~FineGridExchanger();

    virtual void gather();
    virtual void distribute();
    virtual void interpretate();
    virtual void mapBoundary();
    virtual void createMapping();
    virtual void createDataArrays();

    void createBoundary();
    void sendBoundary();
    void setBCFlag();
    void imposeConstraint();
    void imposeBC();

private:

};

#endif

// version
// $Id: FineGridExchanger.h,v 1.17 2003/10/24 04:51:53 tan2 Exp $

// End of file

