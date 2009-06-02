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
#include "PInlet.h"

using Exchanger::Array2D;
using Exchanger::BoundedMesh;
using Exchanger::Sink;

extern "C" {
    double global_p_norm2(struct All_variables*,  double **);
}


PInlet::PInlet(const BoundedMesh& boundedMesh,
	       const Sink& sink,
	       All_variables* e) :
    Inlet(boundedMesh, sink),
    E(e),
    p(sink.size())
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;
}


PInlet::~PInlet()
{}


void PInlet::recv()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    sink.recv(p);

    // TODO: convert pressure to non-dimensional value
    //Exchanger::Convertor& convertor = Convertor::instance();
    //convertor.xpressure(t);

    p.print("CitcomS-PInlet-P");
}


void PInlet::impose()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int n = mesh.nodeID(sink.meshNode(i));
	E->P[m][n] = p[0][i];
    }

    E->monitor.pdotp = global_p_norm2(E, E->P);
}


// private functions


// version
// $Id$

// End of file
