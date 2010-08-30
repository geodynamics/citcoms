// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//
// CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
// Copyright (C) 2002-2005, California Institute of Technology.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//</LicenseText>
//
// Role:
//     impose normal velocity and shear traction as velocity b.c.,
//     also impose temperature as temperature b.c.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_BaseSVTInlet_h)
#define pyCitcomSExchanger_BaseSVTInlet_h

#include "Exchanger/Array2D.h"
#include "Exchanger/DIM.h"
#include "Exchanger/Inlet.h"

struct All_variables;
class Boundary;


class BaseSVTInlet : public Exchanger::Inlet{
protected:
    All_variables* E;
    Exchanger::Array2D<double,Exchanger::STRESS_DIM> s;
    Exchanger::Array2D<double,Exchanger::STRESS_DIM> s_old;
    Exchanger::Array2D<double,Exchanger::DIM> v;
    Exchanger::Array2D<double,Exchanger::DIM> v_old;
    Exchanger::Array2D<double,1> t;
    Exchanger::Array2D<double,1> t_old;

public:
    BaseSVTInlet(const Boundary& boundary,
                 const Exchanger::Sink& sink,
                 All_variables* E);

    virtual ~BaseSVTInlet();

    virtual void impose();

    //virtual void recv() should be implemented by child class

protected:
    void setVBCFlag();
    void setTBCFlag();

    void imposeSV();
    void imposeT();

    double side_tractions(const Exchanger::Array2D<double,Exchanger::STRESS_DIM>& stress,
			  int node, int normal_dir,  int dim) const;

    int nodelevel(int node, int level);
};


#endif

// version
// $Id: BaseSVTInlet.h 2397 2005-10-04 22:37:25Z leif $

// End of file
