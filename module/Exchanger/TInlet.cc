// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//  Purpose:
//  Replace local temperture field by received values. Note that b.c. is not
//  affected.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
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
// $Id: TInlet.cc,v 1.3 2005/06/03 21:51:42 leif Exp $

// End of file
