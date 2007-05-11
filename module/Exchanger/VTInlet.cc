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
// $Id$

// End of file
