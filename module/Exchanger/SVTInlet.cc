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
#include "SVTInlet.h"

extern "C" {

#include "element_definitions.h"

    void construct_side_c3x3matrix_el(const struct All_variables*, int,
				      struct CC*, struct CCX*,
				      int lev,int m,int pressure,int side);
    void get_global_side_1d_shape_fn(const struct All_variables*, int,
				     struct Shape_function1*,
				     struct Shape_function1_dx*,
				     struct Shape_function_side_dA*,
				     int side, int m);
    void check_bc_consistency(const All_variables *E);
    void construct_id(const All_variables *E);
    void temperatures_conform_bcs(All_variables* E);
}


const unsigned vbcFlag[] = {VBX, VBY, VBZ};
const unsigned sbcFlag[] = {SBX, SBY, SBZ};


SVTInlet::SVTInlet(const Boundary& boundary,
		   const Sink& sink,
		   All_variables* E) :
    Inlet(boundary, sink, E)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    s.resize(sink.size());
    s_old.resize(sink.size());

    v.resize(sink.size());
    v_old.resize(sink.size());

    t.resize(sink.size());
    t_old.resize(sink.size());

    setVBCFlag();
    setTBCFlag();

    check_bc_consistency(E);
}


SVTInlet::~SVTInlet()
{}


void SVTInlet::recv()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    // store bc from previous timestep
    s.swap(s_old);
    t.swap(t_old);
    v.swap(v_old);

    sink.recv(t, v);
    sink.recv(s);

    Convertor& convertor = Convertor::instance();
    convertor.xtemperature(t);
    convertor.xvelocity(v, sink.getX());
    //convertor.xstress(s, sink.getX());

    t.print("SVTInlet_T");
    v.print("SVTInlet_V");
    s.print("SVTInlet_S");
}


void SVTInlet::impose()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    //imposeSV();
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
    getFactor(N1, N2);

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int n = boundary.nodeID(sink.meshNode(i));
	for(int d=0; d<DIM; d++)
	    if(E->node[m][n] & vbcFlag[d])
		E->sphere.cap[m].VB[1][n] = N1 * v_old[0][i] + N2 * v[0][i];

	for(int d=0; d<DIM; d++) {
	    int p;
	    if(boundary.normal(d,i) == -1)
		p = sidelow[d];
	    else if(boundary.normal(d,i) == 1)
		p = sidehigh[d];
	    else
		continue;

	    for(int k=0; k<DIM; k++) {
		E->sbc.SB[m][p][k+1][ E->sbc.node[m][n] ] = boundary.normal(d,i) *
		    ( N1 * side_tractions(s_old, i, d, k) +
		      N2 * side_tractions(s, i, d, k) );
	    }
	}
    }
}


void SVTInlet::imposeT()
{
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
// $Id: SVTInlet.cc,v 1.1 2004/04/16 00:03:50 tan2 Exp $

// End of file
