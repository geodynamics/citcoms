// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if !defined(pyCitcom_FineGridExchanger_h)
#define pyCitcom_FineGridExchanger_h


#include "ExchangerClass.h"


class FineGridExchanger : public Exchanger {

public:
    FineGridExchanger(MPI_Comm communicator,
		      MPI_Comm intercomm,
		      int localLeader,
		      int remoteLeader,
		      const All_variables *E);
    virtual ~FineGridExchanger();

    void set_target(const MPI_Comm comm,
			    const MPI_Comm intercomm,
			    const int receiver);

    //virtual void send(int size);
    //virtual void receive(const int size);

    virtual void gather();
    virtual void distribute();
    virtual void interpretate(); // interpolation or extrapolation
    virtual void impose_bc();


};

#endif

// version
// $Id: FineGridExchanger.h,v 1.2 2003/09/08 21:47:27 tan2 Exp $

// End of file

