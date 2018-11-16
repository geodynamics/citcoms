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

#include "config.h"
#include "global_defs.h"
#include "journal/diagnostics.h"
#include "CitcomSource.h"
#include "Convertor.h"
#include "VTOutlet.h"


VTOutlet::VTOutlet(const CitcomSource& source,
		   All_variables* e) :
    Outlet(source),
    E(e),
    v(source.size()),
    t(source.size())

{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;
}


VTOutlet::~VTOutlet()
{}


void VTOutlet::send()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    source.interpolateVelocity(v);
    v.print("CitcomS-VTOutlet-V");

    source.interpolateTemperature(t);
    t.print("CitcomS-VTOutlet-T");

    Exchanger::Convertor& convertor = Convertor::instance();
    convertor.velocity(v, source.getX());
    convertor.temperature(t);

    source.send(t, v);
}


// private functions



// version
// $Id: VTOutlet.cc 6851 2007-05-11 02:02:00Z leif $

// End of file
