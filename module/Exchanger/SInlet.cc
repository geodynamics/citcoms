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
#include "SInlet.h"

extern "C" {

#include "element_definitions.h"

    void check_bc_consistency(const All_variables *E);
    void construct_id(const All_variables *E);
}

using Exchanger::Array2D;
using Exchanger::DIM;
using Exchanger::STRESS_DIM;
using Exchanger::Sink;


const unsigned vbcFlag[] = {VBX, VBY, VBZ};
const unsigned sbcFlag[] = {SBX, SBY, SBZ};


SInlet::SInlet(const Boundary& boundary,
		   const Sink& sink,
		   All_variables* e) :
    Inlet(boundary, sink),
    E(e),
    s(sink.size()),
    s_old(sink.size())
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    //setSBCFlag();

}


SInlet::~SInlet()
{}


void SInlet::recv()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    // store bc from previous timestep
    s.swap(s_old);

    sink.recv(s);

    Exchanger::Convertor& convertor = Convertor::instance();
    convertor.xstress(s, sink.getX());

    s.print("CitcomS-SInlet-S");
}


void SInlet::impose()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    imposeS();
}


// private functions
void SInlet::setSBCFlag()
{
    // BC: normal velocity and shear traction

    const Boundary& boundary = dynamic_cast<const Boundary&>(mesh);
    const int m = 1;

    for(int i=0; i<boundary.size(); i++) {
        int n = boundary.nodeID(i);
		for(int d=0; d<DIM; ++d) {
#if 0
			if(boundary.normal(d,i)) {
				E->node[m][n] = E->node[m][n] | vbcFlag[d];
				E->node[m][n] = E->node[m][n] & (~sbcFlag[d]);
			}
			else {
				E->node[m][n] = E->node[m][n] | sbcFlag[d];
				E->node[m][n] = E->node[m][n] & (~vbcFlag[d]);
			}
#endif
			E->node[m][n] = E->node[m][n] | sbcFlag[d];
			E->node[m][n] = E->node[m][n] & (~vbcFlag[d]);
		}
    }

    // reconstruct ID array to reflect changes in VBC
    construct_id(E);
}


void SInlet::imposeS()
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

		if(E->node[m][n] & (SBX | SBY | SBZ))
			for(int d=0; d<DIM; d++) {
				int p;
				if(boundary.normal(d,j) == -1)
					p = sidelow[d];
				else if(boundary.normal(d,j) == 1)
					p = sidehigh[d];
				else
					continue;

				//	if( side == SIDE_TOP && E->parallel.me_loc[3]==E->parallel.nprocz-1 && (el%E->lmesh.elz==0)) {
				for(int k=0; k<DIM; k++) {
					E->sbc.SB[m][p][k+1][q] =
						boundary.normal(d,j) *
						( N1 * side_tractions(s_old, i, d, k) +
						  N2 * side_tractions(s, i, d, k) );
				}
			}
    }
}


double SInlet::side_tractions(const Array2D<double,STRESS_DIM>& stress,
				int node, int normal_dir, int dim) const
{
    const int stress_index[3][3] =
	{ {0, 3, 4},
	  {3, 1, 5},
	  {4, 5, 2} };

    return stress[ stress_index[normal_dir][dim] ][node];
}


// version
// $Id: SInlet.cc,v 1.2 2005/02/04 18:51:16 ces74 Exp $

// End of file
