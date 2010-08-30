// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//<LicenseText>
//
// CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
// Copyright (C) 2002-2005, California Institute of Technology.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//</LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include "config.h"
#include "journal/diagnostics.h"
#include "global_defs.h"
#include "Boundary.h"
#include "Convertor.h"
#include "Exchanger/Sink.h"
#include "BaseSVTInlet.h"

extern "C" {

#include "element_definitions.h"

    void check_bc_consistency(const All_variables *E);
    void get_bcs_id_for_residual(const All_variables *E, int level, int m);
}

using Exchanger::Array2D;
using Exchanger::DIM;
using Exchanger::STRESS_DIM;
using Exchanger::Sink;


const unsigned vbcFlag[] = {VBX, VBY, VBZ};
const unsigned sbcFlag[] = {SBX, SBY, SBZ};


BaseSVTInlet::BaseSVTInlet(const Boundary& boundary,
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
    debug << journal::at(__HERE__) << journal::endl;

    // set CitcomS boundary flags
    setVBCFlag();
    setTBCFlag();

    check_bc_consistency(E);
}


BaseSVTInlet::~BaseSVTInlet()
{}


void BaseSVTInlet::impose()
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    // impose normal velocity and shear stress as BC for momentum eqn.
    imposeSV();

    // impose temperature BC for energy eqn.
    imposeT();
}


// private functions

void BaseSVTInlet::setVBCFlag()
{
    // BC: normal velocity and shear traction

    const Boundary& boundary = dynamic_cast<const Boundary&>(mesh);
    const int m = 1;

    for(int i=0; i<boundary.size(); i++) {
        int n = boundary.nodeID(i);
        for(int lev=E->mesh.gridmax; lev>=E->mesh.gridmin; lev--) {
            int nn = nodelevel(n, lev);
            if(nn != 0) {
                for(int d=0; d<DIM; ++d)
                    if(boundary.normal(d,i)) {
                        E->NODE[lev][m][nn] = E->NODE[lev][m][nn] | vbcFlag[d];
                        E->NODE[lev][m][nn] = E->NODE[lev][m][nn] & (~sbcFlag[d]);
                    } else {
                        E->NODE[lev][m][nn] = E->NODE[lev][m][nn] | sbcFlag[d];
                        E->NODE[lev][m][nn] = E->NODE[lev][m][nn] & (~vbcFlag[d]);
                    }
            }
        }
    }

    // reconstruct zero_resid array to reflect changes in VBC
    for(int lev=E->mesh.gridmax; lev>=E->mesh.gridmin; lev--)
        get_bcs_id_for_residual(E,lev,m);
}


void BaseSVTInlet::setTBCFlag()
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


void BaseSVTInlet::imposeSV()
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


void BaseSVTInlet::imposeT()
{
    journal::debug_t debugBC("CitcomS-imposeT");
    debugBC << journal::at(__HERE__);

    double N1, N2;
    getTimeFactors(N1, N2);

    const int m = 1;
    for(int i=0; i<sink.size(); i++) {
	int j = sink.meshNode(i);
	int n = mesh.nodeID(j);

	for(int d=0; d<DIM; d++)
	    E->sphere.cap[m].TB[d+1][n] = N1 * t_old[0][i] + N2 * t[0][i];

 	debugBC << E->sphere.cap[m].TB[1][n] << " "
 		<< E->sphere.cap[m].TB[2][n] << " "
 		<< E->sphere.cap[m].TB[3][n] << journal::newline;

    }
    debugBC << journal::endl;

    (E->temperatures_conform_bcs)(E);
}


double BaseSVTInlet::side_tractions(const Array2D<double,STRESS_DIM>& stress,
				int node, int normal_dir, int dim) const
{
    const int stress_index[3][3] =
	{ {0, 3, 4},
	  {3, 1, 5},
	  {4, 5, 2} };

    return stress[ stress_index[normal_dir][dim] ][node];
}


int BaseSVTInlet::nodelevel(int node, int level)
{
    // Given the "node" number of highest level, return the node number
    // at "level" below. Return 0 if this node is not below to lower level.
    const int nox = E->lmesh.nox;
    const int noy = E->lmesh.noy;
    const int noz = E->lmesh.noz;
    const int nx = E->lmesh.NOX[level];
    const int ny = E->lmesh.NOY[level];
    const int nz = E->lmesh.NOZ[level];
    const int f = pow(2, (E->mesh.gridmax - level));

    int i, j, k, ii, jj, kk;

    // indices at highest level, starting from 0
    k = (node - 1) % noz;
    j = (node - 1) / (nox * noz);
    i = ((node - 1) - j * nox * noz) / noz;

    if((i % f) != 0 || (j % f) != 0 || (k % f) != 0) return 0;

    ii = i / f;
    jj = j / f;
    kk = k / f;

    return jj*nx*nz + ii*nz + kk + 1;
}

// version
// $Id: BaseSVTInlet.cc 7643 2007-07-11 20:17:32Z tan2 $

// End of file
