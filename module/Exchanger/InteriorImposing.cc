// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "global_defs.h"
#include "Interior.h"
#include "Sink.h"
#include "Source.h"
#include "InteriorImposing.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


InteriorImposingSink::InteriorImposingSink(const Interior& i, const Sink& s,
					     All_variables* e) :
    E(e),
    interior(i),
    sink(s),
    tic(sink.size())
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in InteriorImposingSink::c'tor"
	  << " sink.size = " << sink.size()
	  << journal::end;
}


void InteriorImposingSink::recvT()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in InteriorImposingSink::recvT" << journal::end;

    // store bc from previous timestep

    sink.recvArray2D(tic);
    tic.print("TIC");

}


void InteriorImposingSink::imposeIC()
{
    imposeTIC();
}


// private functions

void InteriorImposingSink::imposeTIC()
{

    journal::debug_t debugIC("imposeTIC");
    debugIC << journal::loc(__HERE__);

    const int mm = 1;

    for(int i=0; i<sink.size(); i++) {
	int n = sink.meshNode(i);
	for(int d=0; d<DIM; d++)
	    E->T[mm][n] = tic[0][i];
	debugIC << E->T[mm][n] << journal::newline;
    }
    debugIC << journal::end;
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


InteriorImposingSource::InteriorImposingSource(const Source& s,
						 All_variables* e) :
    E(e),
    source(s),
    tic(source.size())
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in InteriorImposingSource::c'tor"
	  << " source.size = " << source.size()
	  << journal::end;
}


void InteriorImposingSource::sendT()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in InteriorImposingSource::sendTandV" << journal::end;

    source.interpolateT(tic, E);
    //tbc.print("TBC");

    source.sendArray2D(tic);
}


// version
// $Id: InteriorImposing.cc,v 1.2 2003/11/07 21:43:47 puru Exp $

// End of file
