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
#include "Exchanger/Sink.h"
#include "SVTInlet.h"

extern "C" {

#include "element_definitions.h"

    void check_bc_consistency(const All_variables *E);
    void construct_id(const All_variables *E);
    void temperatures_conform_bcs(All_variables* E);
}

using Exchanger::Array2D;
using Exchanger::DIM;
using Exchanger::STRESS_DIM;
using Exchanger::Sink;


const unsigned vbcFlag[] = {VBX, VBY, VBZ};
const unsigned sbcFlag[] = {SBX, SBY, SBZ};


SVTInlet::SVTInlet(const Boundary& boundary,
		   const Sink& sink,
		   All_variables* e) :
    Inlet(boundary, sink),
    E(e),
    s(sink.size()),
    s_old(sink.size()),
    v(sink.size()),
    v_old(sink.size()),
    t(sink.size()),
    t_old(sink.size())
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    setVBCFlag();

    check_bc_consistency(E);
}


SVTInlet::~SVTInlet()
{}


void SVTInlet::recv()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    // store bc from previous timestep
    s.swap(s_old);
    t.swap(t_old);
    v.swap(v_old);

    sink.recv(t, v);
    sink.recv(s);

    Exchanger::Convertor& convertor = Convertor::instance();
    convertor.xtemperature(t);
    convertor.xvelocity(v, sink.getX());
    //convertor.xstress(s, sink.getX());

    t.print("CitcomS-SVTInlet-T");
    v.print("CitcomS-SVTInlet-V");
    s.print("CitcomS-SVTInlet-S");
}


void SVTInlet::impose()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    imposeSV();
    imposeT();
}


// private functions

void SVTInlet::setVBCFlag()
{
    // BC: normal velocity and shear traction

    const Boundary& boundary = dynamic_cast<const Boundary&>(mesh);
    const int m = 1;

    for(int i=0; i<boundary.size(); i++) {
        int n = boundary.nodeID(i);
	for(int d=0; d<DIM; ++d)
	    if(boundary.normal(d,i)) {
		E->node[m][n] = E->node[m][n] | vbcFlag[d];
		E->node[m][n] = E->node[m][n] & (~sbcFlag[d]);
	    } else {
		E->node[m][n] = E->node[m][n] | sbcFlag[d];
		E->node[m][n] = E->node[m][n] & (~vbcFlag[d]);
	    }
    }

    // reconstruct ID array to reflect changes in VBC
    construct_id(E);
}


void SVTInlet::setTBCFlag()
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


void SVTInlet::imposeSV()
{
    const int sidelow[3] = {SIDE_NORTH, SIDE_WEST, SIDE_BOTTOM};
    const int sidehigh[3] = {SIDE_SOUTH, SIDE_EAST, SIDE_TOP};

    const Boundary& boundary = dynamic_cast<const Boundary&>(mesh);

    double N1, N2;
    getTimeFactors(N1, N2);

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int j = sink.meshNode(i);
	int n = boundary.nodeID(j);
	int q = E->sbc.node[m][n];

	for(int d=0; d<DIM; d++)
	    if(E->node[m][n] & vbcFlag[d])
		E->sphere.cap[m].VB[d+1][n] = N1 * v_old[d][i] + N2 * v[d][i];

	if(E->node[m][n] & (SBX | SBY | SBZ))
	    for(int d=0; d<DIM; d++) {
		int p;
		if(boundary.normal(d,j) == -1)
		    p = sidelow[d];
		else if(boundary.normal(d,j) == 1)
		    p = sidehigh[d];
		else
		    continue;

		for(int k=0; k<DIM; k++) {
		    E->sbc.SB[m][p][k+1][q] =
			boundary.normal(d,j) *
			( N1 * side_tractions(s_old, i, d, k) +
			  N2 * side_tractions(s, i, d, k) );
		}
	}
    }
}


void SVTInlet::imposeT()
{
    journal::debug_t debugBC("CitcomS-imposeT");
    debugBC << journal::loc(__HERE__);

    const Boundary& boundary = dynamic_cast<const Boundary&>(mesh);
    double N1, N2;
    getTimeFactors(N1, N2);

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int j = sink.meshNode(i);
	int n = mesh.nodeID(j);

	bool influx = false;
	for(int d=0; d<DIM; d++)
	    if( (boundary.normal(d,j) * (N1 * v_old[d][i] + N2 * v[d][i])) < 0 ) {
		influx = true;
		break;
	    }

	if(influx) {
	    E->node[m][n] = E->node[m][n] | TBX;
	    E->node[m][n] = E->node[m][n] | TBY;
	    E->node[m][n] = E->node[m][n] | TBZ;
	    E->node[m][n] = E->node[m][n] & (~FBX);
	    E->node[m][n] = E->node[m][n] & (~FBY);
	    E->node[m][n] = E->node[m][n] & (~FBZ);

	    for(int d=0; d<DIM; d++)
		E->sphere.cap[m].TB[d+1][n] = N1 * t_old[0][i] + N2 * t[0][i];
	}
	else {
	    E->node[m][n] = E->node[m][n] | FBX;
	    E->node[m][n] = E->node[m][n] | FBY;
	    E->node[m][n] = E->node[m][n] | FBZ;
	    E->node[m][n] = E->node[m][n] & (~TBX);
	    E->node[m][n] = E->node[m][n] & (~TBY);
	    E->node[m][n] = E->node[m][n] & (~TBZ);

	    for(int d=0; d<DIM; d++)
		E->sphere.cap[m].TB[d+1][n] = 0;
	}

  	debugBC << E->sphere.cap[m].TB[1][n] << " "
  		<< E->sphere.cap[m].TB[2][n] << " "
  		<< E->sphere.cap[m].TB[3][n] << journal::newline;

    }
    debugBC << journal::end;

    temperatures_conform_bcs(E);
}


double SVTInlet::side_tractions(const Array2D<double,STRESS_DIM>& stress,
				int node, int normal_dir, int dim) const
{
    const int stress_index[3][3] =
	{ {0, 3, 4},
	  {3, 1, 5},
	  {4, 5, 2} };

    return stress[ stress_index[normal_dir][dim] ][node];
}


// version
// $Id: SVTInlet.cc,v 1.4 2004/05/11 07:55:30 tan2 Exp $

// End of file
