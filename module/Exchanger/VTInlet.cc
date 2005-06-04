// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include "journal/diagnostics.h"
#include "global_defs.h"
#include "Convertor.h"
#include "Exchanger/BoundedMesh.h"
#include "Exchanger/Sink.h"
#include "VTInlet.h"

extern "C" {
    void check_bc_consistency(const All_variables *E);
    void construct_id(const All_variables *E);
}

using Exchanger::Array2D;
using Exchanger::BoundedMesh;
using Exchanger::DIM;
using Exchanger::Sink;


VTInlet::VTInlet(const BoundedMesh& boundedMesh,
		 const Sink& sink,
		 All_variables* e) :
    Inlet(boundedMesh, sink),
    E(e),
    v(sink.size()),
    v_old(sink.size()),
    t(sink.size()),
    t_old(sink.size())
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    setVBCFlag();
    setTBCFlag();

    check_bc_consistency(E);
}


VTInlet::~VTInlet()
{}


void VTInlet::recv()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    // store bc from previous timestep
    t.swap(t_old);
    v.swap(v_old);

    sink.recv(t, v);

    Exchanger::Convertor& convertor = Convertor::instance();
    convertor.xtemperature(t);
    convertor.xvelocity(v, sink.getX());

    t.print("CitcomS-VTInlet-T");
    v.print("CitcomS-VTInlet-V");
}


void VTInlet::impose()
{
    imposeV();
    imposeT();
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


void VTInlet::imposeV()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    journal::debug_t debugBC("CitcomS-VTInlet-imposeV");
    debugBC << journal::at(__HERE__);

    double N1, N2;
    getTimeFactors(N1, N2);

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
    debugBC << journal::endl;
}


void VTInlet::imposeT()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    journal::debug_t debugBC("CitcomS-VTInlet-imposeT");
    debugBC << journal::at(__HERE__);

    double N1, N2;
    getTimeFactors(N1, N2);

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
    debugBC << journal::endl;

    (E->temperatures_conform_bcs)(E);
}


// version
// $Id: VTInlet.cc,v 1.7 2005/06/03 21:51:42 leif Exp $

// End of file
