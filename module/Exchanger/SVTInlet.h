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

#if !defined(pyCitcom_SVTInlet_h)
#define pyCitcom_SVTInlet_h

#include <string>
#include "Array2D.h"
#include "DIM.h"
#include "Inlet.h"

class Boundary;
class Sink;


class SVTInlet : public Inlet{
protected:
    Array2D<double,STRESS_DIM> s;
    Array2D<double,STRESS_DIM> s_old;
    Array2D<double,DIM> v;
    Array2D<double,DIM> v_old;
    Array2D<double,1> t;
    Array2D<double,1> t_old;

public:
    SVTInlet(const Boundary& boundary,
	     const Sink& sink,
	     All_variables* E);

    virtual ~SVTInlet();

    virtual void recv();
    virtual void impose();

private:
    void setVBCFlag();
    void setTBCFlag();

    void imposeSV();
    void imposeT();

    double side_tractions(const Array2D<double,STRESS_DIM>& stress,
			  int node, int normal_dir,  int dim) const;
};


#endif

// version
// $Id: SVTInlet.h,v 1.1 2004/04/16 00:03:50 tan2 Exp $

// End of file
