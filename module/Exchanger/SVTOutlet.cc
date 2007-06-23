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
#include "journal/diagnostics.h"
#include "Convertor.h"
#include "CitcomSource.h"
#include "SVTOutlet.h"

using Exchanger::Array2D;
using Exchanger::DIM;
using Exchanger::STRESS_DIM;


SVTOutlet::SVTOutlet(const CitcomSource& source,
		     All_variables* e) :
    Outlet(source),
    E(e),
    s(source.size()),
    v(source.size()),
    t(source.size())
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;
}


SVTOutlet::~SVTOutlet()
{}


void SVTOutlet::send()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    source.interpolateStress(s);
    s.print("CitcomS-SVTOutlet-S");

    source.interpolateVelocity(v);
    v.print("CitcomS-SVTOutlet-V");

    source.interpolateTemperature(t);
    t.print("CitcomS-SVTOutlet-T");

    Exchanger::Convertor& convertor = Convertor::instance();
    //convertor.stress(s, source.getX());
    convertor.velocity(v, source.getX());
    convertor.temperature(t);

    source.send(t, v);
    source.send(s);
}


// version
// $Id$

// End of file
