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
#include "BoundedMesh.h"
#include "Convertor.h"
#include "Sink.h"
#include "VTInlet.h"

extern "C" {
    void check_bc_consistency(const All_variables *E);
    void construct_id(const All_variables *E);
    void temperatures_conform_bcs(All_variables* E);
}


VTInlet::VTInlet(const BoundedMesh& boundedMesh,
		 const Sink& sink,
		 All_variables* E,
		 const std::string& mode) :
    Inlet(boundedMesh, sink, E),
    modeV(mode.find('V',0) != std::string::npos),
    modeT(mode.find('T',0) != std::string::npos),
    modet(mode.find('t',0) != std::string::npos)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "modeV = " << modeV << "  modeT = " << modeT
	  << "  modet= " << modet << journal::end;

    if((modeT && modet) || !(modeV || modeT || modet)) {
	journal::firewall_t firewall("VTInlet");
	firewall << journal::loc(__HERE__)
		 << "invalid mode" << journal::end;
    }

    if(modeV) {
	v.resize(sink.size());
	v_old.resize(sink.size());
	setVBCFlag();
    }

    if(modeT || modet) {
	t.resize(sink.size());
	t_old.resize(sink.size());
    }

    if(modeT)
	setTBCFlag();

    check_bc_consistency(E);
}


VTInlet::~VTInlet()
{}


void VTInlet::recv()
{
    // store bc from previous timestep
    t.swap(t_old);
    v.swap(v_old);

    if(modeV && (modeT || modet))
	recvVT();
    else if(modeV)
	recvV();
    else
	recvT();
}


void VTInlet::impose()
{
    if(modeV)
 	imposeV();

    if(modeT)
 	imposeT();

    if(modet)
 	imposet();
}


// private functions

void VTInlet::setVBCFlag()
{
    // Because CitcomS' default side BC is reflecting,
    // here we should change to velocity BC.
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

    // reconstruct ID array to reflect changes in VBC
    construct_id(E);
}


void VTInlet::setTBCFlag()
{
    const int m = 1;
    for(int i=0; i<mesh.size(); i++) {
	int n = mesh.nodeID(i);
	E->node[m][n] = E->node[m][n] | TBX;
	E->node[m][n] = E->node[m][n] | TBY;
	E->node[m][n] = E->node[m][n] | TBZ;
	E->node[m][n] = E->node[m][n] & (~FBX);
	E->node[m][n] = E->node[m][n] & (~FBY);
	E->node[m][n] = E->node[m][n] & (~FBZ);
    }
}


void VTInlet::recvVT()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    sink.recv(t, v);

    Convertor& convertor = Convertor::instance();
    convertor.xtemperature(t);
    convertor.xvelocity(v, sink.getX());

    t.print("VTInlet_T");
    v.print("VTInlet_V");
}


void VTInlet::recvV()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    sink.recvD(v);

    Convertor& convertor = Convertor::instance();
    convertor.xvelocity(v, sink.getX());

    v.print("VTInlet_V");
}


void VTInlet::recvT()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    sink.recv(t);

    Convertor& convertor = Convertor::instance();
    convertor.xtemperature(t);

    t.print("VTInlet_T");
}


void VTInlet::imposeV()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    journal::debug_t debugBC("imposeV");
    debugBC << journal::loc(__HERE__);

    double N1, N2;
    getFactor(N1, N2);

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int n = mesh.nodeID(sink.meshNode(i));
	for(int d=0; d<DIM; d++)
	    E->sphere.cap[m].VB[d+1][n] = N1 * v_old[d][i]
		                        + N2 * v[d][i];

 	debugBC << E->sphere.cap[m].VB[1][n] << " "
 		<< E->sphere.cap[m].VB[2][n] << " "
 		<< E->sphere.cap[m].VB[3][n] << journal::newline;
    }
    debugBC << journal::end;
}


void VTInlet::imposeT()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    journal::debug_t debugBC("imposeT");
    debugBC << journal::loc(__HERE__);

    double N1, N2;
    getFactor(N1, N2);

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int n = mesh.nodeID(sink.meshNode(i));
 	for(int d=0; d<DIM; d++)
 	    E->sphere.cap[m].TB[d+1][n] = N1 * t_old[0][i]
 		                        + N2 * t[0][i];

  	debugBC << E->sphere.cap[m].TB[1][n] << " "
  		<< E->sphere.cap[m].TB[2][n] << " "
  		<< E->sphere.cap[m].TB[3][n] << journal::newline;

    }
    debugBC << journal::end;

    temperatures_conform_bcs(E);
}


void VTInlet::imposet()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    journal::debug_t debugBC("imposet");
    debugBC << journal::loc(__HERE__);

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int n = mesh.nodeID(sink.meshNode(i));
	E->T[m][n] = t[0][i];

  	debugBC << E->T[m][n] << journal::newline;
    }
    debugBC << journal::end;

    temperatures_conform_bcs(E);
}


// version
// $Id: VTInlet.cc,v 1.3 2004/03/11 22:36:47 tan2 Exp $

// End of file
