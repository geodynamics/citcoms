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
#include "journal/journal.h"
#include "global_defs.h"
#include "Convertor.h"
#include "Exchanger/BoundedMesh.h"
#include "Exchanger/Sink.h"
#include "TInlet.h"

extern "C" {
    void temperatures_conform_bcs(All_variables* E);
}

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
    debug << journal::loc(__HERE__) << journal::end;
}


TInlet::~TInlet()
{}


void TInlet::recv()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    sink.recv(t);

    Exchanger::Convertor& convertor = Convertor::instance();
    convertor.xtemperature(t);

    t.print("CitcomS-TInlet-T");
}


void TInlet::impose()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int n = mesh.nodeID(sink.meshNode(i));
	E->T[m][n] = t[0][i];
    }

    temperatures_conform_bcs(E);
}


// private functions


// version
// $Id: TInlet.cc,v 1.1 2004/05/11 07:55:30 tan2 Exp $

// End of file
