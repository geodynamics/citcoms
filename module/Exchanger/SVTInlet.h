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

#if !defined(pyCitcomSExchanger_SVTInlet_h)
#define pyCitcomSExchanger_SVTInlet_h

#include "Exchanger/Array2D.h"
#include "Exchanger/DIM.h"
#include "Exchanger/Inlet.h"

struct All_variables;
class Boundary;


class SVTInlet : public Exchanger::Inlet{
    All_variables* E;
    Exchanger::Array2D<double,Exchanger::STRESS_DIM> s;
    Exchanger::Array2D<double,Exchanger::STRESS_DIM> s_old;
    Exchanger::Array2D<double,Exchanger::DIM> v;
    Exchanger::Array2D<double,Exchanger::DIM> v_old;
    Exchanger::Array2D<double,1> t;
    Exchanger::Array2D<double,1> t_old;

public:
    SVTInlet(const Boundary& boundary,
	     const Exchanger::Sink& sink,
	     All_variables* E);

    virtual ~SVTInlet();

    virtual void recv();
    virtual void impose();

private:
    void setVBCFlag();
    void setTBCFlag();

    void imposeSV();
    void imposeT();

    double side_tractions(const Exchanger::Array2D<double,Exchanger::STRESS_DIM>& stress,
			  int node, int normal_dir,  int dim) const;
};


#endif

// version
// $Id: SVTInlet.h,v 1.2 2004/05/11 07:55:30 tan2 Exp $

// End of file
