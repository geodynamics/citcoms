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
#include "Convertor.h"
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
	  << "in InteriorImposingSink::Citcom"
	  << " sink.size = " << sink.size()
	  << journal::end;
}


void InteriorImposingSink::recvT()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in InteriorImposingSink::recvT" << journal::end;

    sink.recvArray2D(tic);

    Convertor& convertor = Convertor::instance();
    convertor.xtemperature(tic);

    tic.print("TIC");

}

void InteriorImposingSink::recvV()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in InteriorImposingSink::recvV" << journal::end;
    
    
    sink.recvArray2D(vic);

    Convertor& convertor = Convertor::instance();
    convertor.xvelocity(vic,sink.getX());

    vic.print("VIC");
    imposeVIC();
    
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
	int n = interior.nodeID(sink.meshNode(i));
	E->T[mm][n] = tic[0][i];
	debugIC << E->T[mm][n] << journal::newline;
    }
    debugIC << journal::end;
}

void InteriorImposingSink::imposeVIC()
{
    journal::debug_t debugVIC("imposeVIC");
    debugVIC << journal::loc(__HERE__);

    const int mm = 1;

    for(int i=0; i<sink.size(); i++) {
	int n = interior.nodeID(sink.meshNode(i));
        for(int d=0; d<DIM; d++)
          E->sphere.cap[mm].VB[d+1][n] =vic[d][i];  
        
	debugVIC << E->T[mm][n] << journal::newline;
    }
    debugVIC << journal::end;
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
	  << "in InteriorImposingSource::Citcom"
	  << " source.size = " << source.size()
	  << journal::end;
}


void InteriorImposingSource::sendT()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in InteriorImposingSource::sendT" << journal::end;

    source.interpolateT(tic, E);
    tic.print("TIC");

    Convertor& convertor = Convertor::instance();
    convertor.temperature(tic);

    source.sendArray2D(tic);
}


// version
// $Id: InteriorImposing.cc,v 1.13 2004/01/21 00:13:53 puru Exp $

// End of file
