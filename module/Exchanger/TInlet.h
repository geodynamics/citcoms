// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
// Role:
//     impose velocity and temperature as b.c.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_TInlet_h)
#define pyCitcomSExchanger_TInlet_h

#include "Exchanger/Array2D.h"
#include "Exchanger/DIM.h"
#include "Exchanger/Inlet.h"

struct All_variables;


class TInlet : public Exchanger::Inlet {
    All_variables* E;
    Exchanger::Array2D<double,1> t;

public:
    TInlet(const Exchanger::BoundedMesh& boundedMesh,
	   const Exchanger::Sink& sink,
	   All_variables* E);

    virtual ~TInlet();

    virtual void recv();
    virtual void impose();

private:

};


#endif

// version
// $Id: TInlet.h,v 1.1 2004/05/11 07:55:30 tan2 Exp $

// End of file
