// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "journal/journal.h"
#include "global_defs.h"
#include "Boundary.h"
#include "Convertor.h"
#include "Sink.h"
#include "TractionInlet.h"

extern "C" {
    void check_bc_consistency(const All_variables *E);
    void construct_id(const All_variables *E);
    void temperatures_conform_bcs(All_variables* E);
}


TractionInlet::TractionInlet(const Boundary& boundary,
			     const Sink& sink,
			     All_variables* E,
			     const std::string& mode) :
    Inlet(boundary, sink, E),
    modeF(mode.find('F',0) != std::string::npos),
    modeV(mode.find('V',0) != std::string::npos)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
          << "modeF = " << modeF << "  modeV = " << modeV
	  << journal::end;

    if(modeF) {
        f.resize(sink.size());
        f_old.resize(sink.size());
    }

    if(modeV) {
        v.resize(sink.size());
        v_old.resize(sink.size());
    }

    setVBCFlag();
    check_bc_consistency(E);
}


TractionInlet::~TractionInlet()
{}


void TractionInlet::recv()
{
    // store bc from previous timestep
    f.swap(f_old);
    v.swap(v_old);

    if(modeV && modeF)
        recvFV();
    else if(modeV)
        recvV();
    else
        recvF();
}


void TractionInlet::impose()
{
    if(modeV && modeF)
	imposeFV();
    else if(modeV)
        imposeV();
    else
        imposeF();

}


// private functions

void TractionInlet::setVBCFlag()
{
    if(modeV && modeF)
	{}
// 	setMixedBC();
    else if(modeV)
        setVBC();
    else
	setFBC();

    // reconstruct ID array to reflect changes in VBC
    construct_id(E);
}


void TractionInlet::setMixedBC()
{
    // BC: normal velocity and shear traction

    const Boundary& boundary = dynamic_cast<const Boundary&>(mesh);
    const int m = 1;

    for(int i=0; i<boundary.size(); i++) {
        int n = boundary.nodeID(i);

	if(boundary.normal(0,i)) {
	    E->node[m][n] = E->node[m][n] | VBX;
	    E->node[m][n] = E->node[m][n] & (~SBX);
	} else {
	    E->node[m][n] = E->node[m][n] | SBX;
	    E->node[m][n] = E->node[m][n] & (~VBX);
	}

	if(boundary.normal(1,i)) {
	    E->node[m][n] = E->node[m][n] | VBY;
	    E->node[m][n] = E->node[m][n] & (~SBY);
	} else {
	    E->node[m][n] = E->node[m][n] | SBY;
	    E->node[m][n] = E->node[m][n] & (~VBY);
	}

	if(boundary.normal(2,i)) {
	    E->node[m][n] = E->node[m][n] | VBZ;
	    E->node[m][n] = E->node[m][n] & (~SBZ);
	} else {
	    E->node[m][n] = E->node[m][n] | SBZ;
	    E->node[m][n] = E->node[m][n] & (~VBZ);
	}
    }
}


void TractionInlet::setVBC()
{
    const int m = 1;

    for(int i=0; i<mesh.size(); i++) {
        int n = mesh.nodeID(i);
        E->node[m][n] = E->node[m][n] | VBX;
        E->node[m][n] = E->node[m][n] | VBY;
        E->node[m][n] = E->node[m][n] | VBZ;
        E->node[m][n] = E->node[m][n] & (~SBX);
        E->node[m][n] = E->node[m][n] & (~SBY);
        E->node[m][n] = E->node[m][n] & (~SBZ);
    }
}


void TractionInlet::setFBC()
{
    const int m = 1;

    for(int i=0; i<mesh.size(); i++) {
        int n = mesh.nodeID(i);
        E->node[m][n] = E->node[m][n] | SBX;
        E->node[m][n] = E->node[m][n] | SBY;
        E->node[m][n] = E->node[m][n] | SBZ;
        E->node[m][n] = E->node[m][n] & (~VBX);
        E->node[m][n] = E->node[m][n] & (~VBY);
        E->node[m][n] = E->node[m][n] & (~VBZ);
    }
}


void TractionInlet::recvFV()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    sink.recv(f, v);

    Convertor& convertor = Convertor::instance();
    convertor.xtraction(f, sink.getX());
    convertor.xvelocity(v, sink.getX());

    f.print("TractionInlet_F");
    v.print("TractionInlet_V");
}


void TractionInlet::recvF()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    sink.recv(f);

    Convertor& convertor = Convertor::instance();
    convertor.xtraction(f, sink.getX());

    f.print("TractionInlet_F");
}


void TractionInlet::recvV()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    sink.recv(v);

    Convertor& convertor = Convertor::instance();
    convertor.xvelocity(v, sink.getX());

    v.print("TractionInlet_V");
}


void TractionInlet::imposeFV()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    const Boundary& boundary = dynamic_cast<const Boundary&>(mesh);
    double N1, N2;
    getFactor(N1, N2);

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int k = sink.meshNode(i);
        int n = mesh.nodeID(k);

	if(E->node[m][n] & VBX)
	    E->sphere.cap[m].VB[1][n] = N1 * v_old[0][i] + N2 * v[0][i];
	else
	    E->sphere.cap[m].VB[1][n] = N1 * f_old[0][i] + N2 * f[0][i];

	if(E->node[m][n] & VBY)
	    E->sphere.cap[m].VB[2][n] = N1 * v_old[1][i] + N2 * v[1][i];
	else
	    E->sphere.cap[m].VB[2][n] = N1 * f_old[1][i] + N2 * f[1][i];

	if(E->node[m][n] & VBZ)
	    E->sphere.cap[m].VB[3][n] = N1 * v_old[2][i] + N2 * v[2][i];
	else
	    E->sphere.cap[m].VB[3][n] = N1 * f_old[2][i] + N2 * f[2][i];


//         for(int d=0; d<DIM; d++) {
// 	    if(boundary.normal(d, k))
// 		E->sphere.cap[m].VB[d+1][n] = N1 * v_old[d][i]
//                                             + N2 * v[d][i];
// 	    else
// 		E->sphere.cap[m].VB[d+1][n] = N1 * f_old[d][i]
//                                             + N2 * f[d][i];
// 	}
    }
}


void TractionInlet::imposeF()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    double N1, N2;
    getFactor(N1, N2);

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int k = sink.meshNode(i);
        int n = mesh.nodeID(k);
        for(int d=0; d<DIM; d++)
	    E->sphere.cap[m].VB[d+1][n] = N1 * f_old[d][i]
		                        + N2 * f[d][i];
    }
}


void TractionInlet::imposeV()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    double N1, N2;
    getFactor(N1, N2);

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int k = sink.meshNode(i);
        int n = mesh.nodeID(k);
        for(int d=0; d<DIM; d++)
	    E->sphere.cap[m].VB[d+1][n] = N1 * v_old[d][i]
		                        + N2 * v[d][i];
    }
}


// version
// $Id: TractionInlet.cc,v 1.1 2004/03/28 23:05:19 tan2 Exp $

// End of file
