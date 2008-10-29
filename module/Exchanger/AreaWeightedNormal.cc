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
#include <cmath>
#include <iostream>
#include "global_defs.h"
#include "journal/diagnostics.h"
#include "Boundary.h"
#include "Exchanger/UtilTemplate.h"
#include "Exchanger/Array2D.h"
#include "Exchanger/Sink.h"
#include "AreaWeightedNormal.h"

extern "C" {
#include "element_definitions.h"
}


//using Exchanger::Array2D;
using Exchanger::DIM;
using Exchanger::Sink;


AreaWeightedNormal::AreaWeightedNormal(const MPI_Comm& comm,
				       const Boundary& boundary,
				       const Sink& sink,
				       const All_variables* E) :
    size_(boundary.size()),
    toleranceOutflow_(E->control.accuracy),
    nwght(size_, 0)
{
    computeWeightedNormal(boundary, E);
    double total_area = computeTotalArea(comm, sink);
    normalize(total_area);
}


AreaWeightedNormal::~AreaWeightedNormal()
{}


void AreaWeightedNormal::imposeConstraint(Velo& V,
					  const MPI_Comm& comm,
					  const Sink& sink) const
{
    journal::info_t info("CitcomS-AreaWeightedNormal-outflow");

    double outflow = computeOutflow(V, comm, sink);
    info << journal::at(__HERE__)
	 << "Net outflow: "
	 << outflow << journal::endl;

    if (std::abs(outflow) > toleranceOutflow_) {
	reduceOutflow(V, outflow, sink);

	outflow = computeOutflow(V, comm, sink);
	info << journal::at(__HERE__)
	     << "Net outflow after correction (SHOULD BE ZERO !): "
	     << outflow << journal::endl;
    }
}


// private functions

void AreaWeightedNormal::computeWeightedNormal(const Boundary& boundary,
					       const All_variables* E)
{
    const int nodes_per_element = 8;
    const int vpoints_per_side = 4;
    const int side_normal[DIM][DIM] = {{SIDE_NORTH, -1, SIDE_SOUTH},
				       {SIDE_WEST, -1, SIDE_EAST},
				       {SIDE_BOTTOM, -1, SIDE_TOP}};

    for (int m=1; m<=E->sphere.caps_per_proc; m++)
	for (int es=1; es<=E->boundary.nel; es++) {
	    int el = E->boundary.element[m][es];
	    for(int n=1; n<=nodes_per_element; n++) {
		int node = E->ien[m][el].node[n];
		int bnode = boundary.bnode(node);
		if(bnode < 0) continue;

		for(int j=0; j<DIM; j++) {
		    int normal = boundary.normal(j,bnode);
		    if(normal) {
			int side = side_normal[j][normal+1];
			for(int k=1; k<=vpoints_per_side; k++)
			    nwght[j][bnode] += normal * E->boundary.det[m][side][k][es];
		    }
		} // end of loop over dim
	    } // end of loop over element nodes
	} // end of loop over elements
}


double AreaWeightedNormal::computeTotalArea(const MPI_Comm& comm,
					    const Sink& sink) const
{
    double total_area = 0;
    for(int n=0; n<sink.size(); n++) {
	int i = sink.meshNode(n);
	for(int j=0; j<DIM; j++)
	    total_area += std::abs(nwght[j][i]);
    }

    Exchanger::util::gatherSum(comm, total_area);

    return total_area;
}


void AreaWeightedNormal::normalize(double total_area)
{
    for(int i=0; i<size_; i++)
	for(int j=0; j<DIM; j++)
	    nwght[j][i] /= total_area;
}


double AreaWeightedNormal::computeOutflow(const Velo& V,
					  const MPI_Comm& comm,
					  const Sink& sink) const
{
    double outflow = 0;
    for(int n=0; n<sink.size(); n++) {
	int i = sink.meshNode(n);
	for(int j=0; j<DIM; j++)
	    outflow += V[j][n] * nwght[j][i];
    }

    Exchanger::util::gatherSum(comm, outflow);

    return outflow;
}


int AreaWeightedNormal::sign(double number) const
{
    return (number > 0) ? 1 : ((number < 0) ? -1 : 0);
}


void AreaWeightedNormal::reduceOutflow(Velo& V, double outflow,
				       const Sink& sink) const
{
    for(int n=0; n<sink.size(); n++) {
	int i = sink.meshNode(n);
	for(int j=0; j<DIM; j++)
	    V[j][n] -= sign(nwght[j][i]) * outflow;
    }
}



// version
// $Id$

// End of file
