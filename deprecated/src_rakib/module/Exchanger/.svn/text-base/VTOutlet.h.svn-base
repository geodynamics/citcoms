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
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_VTOutlet_h)
#define pyCitcomSExchanger_VTOutlet_h

#include "Exchanger/Outlet.h"

struct All_variables;
class CitcomSource;


class VTOutlet : public Exchanger::Outlet {
    All_variables* E;
    Exchanger::Array2D<double,Exchanger::DIM> v;
    Exchanger::Array2D<double,1> t;

public:
    VTOutlet(const CitcomSource& source,
	     All_variables* E);
    virtual ~VTOutlet();

    virtual void send();

private:
    // disable copy c'tor and assignment operator
    VTOutlet(const VTOutlet&);
    VTOutlet& operator=(const VTOutlet&);

};


#endif

// version
// $Id$

// End of file
