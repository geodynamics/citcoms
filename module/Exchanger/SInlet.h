// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
// Role:
//     impose normal velocity and shear traction as velocity b.c.,
//     also impose temperature as temperature b.c.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_SInlet_h)
#define pyCitcomSExchanger_SInlet_h

#include "Exchanger/Array2D.h"
#include "Exchanger/DIM.h"
#include "Exchanger/Inlet.h"

struct All_variables;
class Boundary;


class SInlet : public Exchanger::Inlet{
    All_variables* E;
    Exchanger::Array2D<double,Exchanger::STRESS_DIM> s;
    Exchanger::Array2D<double,Exchanger::STRESS_DIM> s_old;

public:
    SInlet(const Boundary& boundary,
	     const Exchanger::Sink& sink,
	     All_variables* E);

    virtual ~SInlet();

    virtual void recv();
    virtual void impose();

private:
    void imposeS();

    double side_tractions(const Exchanger::Array2D<double,Exchanger::STRESS_DIM>& stress,
			  int node, int normal_dir,  int dim) const;
};


#endif

// version
// $Id: SInlet.h,v 1.1 2005/01/29 00:15:57 ces74 Exp $

// End of file
