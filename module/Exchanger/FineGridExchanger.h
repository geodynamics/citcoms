// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if !defined(pyCitcom_FineGridExchanger_h)
#define pyCitcom_FineGridExchanger_h


#include "Exchanger.h"


class FineGridExchanger : public Exchanger {

public:
    FineGridExchanger(const All_variables *E);
    ~FineGridExchanger();

    virtual void set_target(const MPI_Comm comm,
			    const MPI_Comm intercomm,
			    const int receiver);
    virtual void send(int size);
    virtual void receive(const int size);

    virtual void gather();
    virtual void distribute();
    virtual void interpretate(); // interpolation or extrapolation
    virtual void impose_bc();
};

#endif

// version
// $Id: FineGridExchanger.h,v 1.1 2003/09/06 23:44:22 tan2 Exp $

// End of file

