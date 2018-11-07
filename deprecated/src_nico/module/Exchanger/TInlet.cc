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
//  Purpose:
//  Replace local temperture field by received values. Note that b.c. is not
//  affected.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include "config.h"
#include "journal/diagnostics.h"
#include "global_defs.h"
#include "Convertor.h"
#include "Exchanger/BoundedMesh.h"
#include "Exchanger/Sink.h"
#include "TInlet.h"

using Exchanger::Array2D;
using Exchanger::BoundedMesh;
using Exchanger::Sink;


TInlet::TInlet(const BoundedMesh& boundedMesh,
	       const Sink& sink,
	       All_variables* e) :
    Inlet(boundedMesh, sink),
    E(e),
    t(sink.size())
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;
}


TInlet::~TInlet()
{}


void TInlet::recv()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    sink.recv(t);

    Exchanger::Convertor& convertor = Convertor::instance();
    convertor.xtemperature(t);

    t.print("CitcomS-TInlet-T");
}


void TInlet::impose()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int n = mesh.nodeID(sink.meshNode(i));
	E->T[m][n] = t[0][i];
    }

    (E->temperatures_conform_bcs)(E);
}


// private functions


// version
// $Id: TInlet.cc 6851 2007-05-11 02:02:00Z leif $

// End of file
