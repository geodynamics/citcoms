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
    tic(sink.size()),
    vic(sink.size())
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

    sink.recvArray2D(tic);
    tic.print("TIC");

}

void InteriorImposingSink::recvV()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in InteriorImposingSink::recvV" << journal::end;

    sink.recvArray2D(vic);
    vic.print("VIC");

}

void InteriorImposingSink::imposeIC()
{
    imposeTIC();
}

void InteriorImposingSink::imposeVIC()
{
    imposeVICP();
}


// private functions

void InteriorImposingSink::imposeTIC()
{
    journal::debug_t debugIC("imposeTIC");
    debugIC << journal::loc(__HERE__);

    const int mm = 1;

    for(int i=0; i<sink.size(); i++) {
	int n = interior.nodeID(sink.meshNode(i));
	E->T[mm][n] = tic[0][i];
	debugIC << E->T[mm][n] << journal::newline;
    }
    debugIC << journal::end;
}
void InteriorImposingSink::imposeVICP()
{
    journal::debug_t debugVICP("imposeVICP");
    debugVICP << journal::loc(__HERE__);

    const int mm = 1;

    for(int i=0; i<sink.size(); i++) {
	int n = interior.nodeID(sink.meshNode(i));
        for(int d=0; d<DIM; d++)
            E->sphere.cap[mm].VB[d+1][n]=vic[d][i];
        
	debugVICP << E->T[mm][n] << journal::newline;
    }
    debugVICP << journal::end;
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
	  << "in InteriorImposingSource::sendT" << journal::end;

    source.interpolateT(tic, E);
    //tbc.print("TIC");

    source.sendArray2D(tic);
}


// version
// $Id: InteriorImposing.cc,v 1.6 2003/11/25 18:41:16 puru Exp $

// End of file
