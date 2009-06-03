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
#include "global_defs.h"
#include "Boundary.h"
#include "Convertor.h"
#include "Exchanger/Sink.h"
#include "SVTInlet.h"

using Exchanger::Sink;


SVTInlet::SVTInlet(const Boundary& boundary,
		   const Sink& sink,
		   All_variables* e) :
    BaseSVTInlet(boundary, sink, e)
{}


SVTInlet::~SVTInlet()
{}


void SVTInlet::recv()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    // store bc from previous timestep
    s.swap(s_old);
    t.swap(t_old);
    v.swap(v_old);

    // receive the fields from outlet
    sink.recv(t, v);
    sink.recv(s);

    // convert back to CitcomS' units and coordinate system
    Exchanger::Convertor& convertor = Convertor::instance();
    convertor.xtemperature(t);
    convertor.xvelocity(v, sink.getX());
    //convertor.xstress(s, sink.getX());

    t.print("CitcomS-SVTInlet-T");
    v.print("CitcomS-SVTInlet-V");
    s.print("CitcomS-SVTInlet-S");
}


// version
// $Id$

// End of file
