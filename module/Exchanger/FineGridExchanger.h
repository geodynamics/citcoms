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
		      const int localrank,
		      const int interrank,
		      const int localLeader,
		      const int remoteLeader,
		      const All_variables *E);
    virtual ~FineGridExchanger();

  //    virtual void gather();
  //    virtual void distribute();
    virtual void interpretate();
//     virtual void imposeBC();
  
  void mapBoundary();
  void createBoundary();
  void sendBoundary();

};

#endif

// version
// $Id: FineGridExchanger.h,v 1.11 2003/09/27 17:12:52 tan2 Exp $

// End of file

