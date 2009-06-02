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
    // for definition of SIDE_NORTH etc.
#include "element_definitions.h"
}


//using Exchanger::Array2D;
using Exchanger::DIM;
using Exchanger::Sink;


AreaWeightedNormal::AreaWeightedNormal(const MPI_Comm& comm,
				       const Boundary& boundary,
				       const Sink& sink,
				       const All_variables* E) :
    nwght(boundary.size(), 0)
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    computeWeightedNormal(boundary, E);
    double inv_total_area = 1 / computeTotalArea(comm, sink);

    // normalize
    for(int n=0; n<sink.size(); n++) {
	int i = sink.meshNode(n);
	for(int j=0; j<DIM; j++)
	    nwght[j][i] *= inv_total_area;
    }
}


AreaWeightedNormal::~AreaWeightedNormal()
{}


void AreaWeightedNormal::imposeConstraint(Velo& V,
					  const MPI_Comm& comm,
					  const Sink& sink,
					  const All_variables* E) const
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::at(__HERE__) << journal::endl;

    double outflow = computeOutflow(V, comm, sink);

    reduceOutflow(V, outflow, sink);

    double new_outflow = computeOutflow(V, comm, sink);

    if(E->parallel.me == 0) {
        fprintf(stderr, "Net outflow amended from %e to %e\n",
                outflow, new_outflow);
        fprintf(E->fp, "Net outflow amended from %e to %e\n",
                outflow, new_outflow);
    }
}


// private functions

void AreaWeightedNormal::computeWeightedNormal(const Boundary& boundary,
					       const All_variables* E)
{
    const int nodes_per_element = 8;
    const int vpoints_per_side = 4;

    // converting normal vector [-1, 0, 1] to side index,
    const int side_normal[DIM][DIM] = {{SIDE_NORTH, -1, SIDE_SOUTH},
				       {SIDE_WEST, -1, SIDE_EAST},
				       {SIDE_BOTTOM, -1, SIDE_TOP}};

    /* For each node belong to boundary elements, check all 6 faces,
     * if the node is on the face and the face is part of the boundary,
     * compute the surface area by summing the 2D determinants. */
    for (int m=1; m<=E->sphere.caps_per_proc; m++)
	for (int es=1; es<=E->boundary.nel; es++) {
	    int el = E->boundary.element[m][es];
	    for(int n=1; n<=nodes_per_element; n++) {
		int node = E->ien[m][el].node[n];
		int bnode = boundary.bnode(node);

                // skipping nodes not on the boundary
		if(bnode < 0) continue;

		for(int j=0; j<DIM; j++) {
                    // normal is either -1, 1, or 0
		    int normal = boundary.normal(j, bnode);
		    if(normal) {
			int side = side_normal[j][normal+1];
			for(int k=1; k<=vpoints_per_side; k++)
			    nwght[j][bnode] += normal * E->boundary.det[m][side][k][es] / vpoints_per_side;
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

    journal::info_t info("CitcomS-AreaWeightedNormal-area");
    info << journal::at(__HERE__)
         << "Total surface area: "
         << total_area << journal::endl;

    /** debug **
    for(int j=0; j<DIM; j++) {
        double in, out;
        in = out = 0;
        for(int n=0; n<sink.size(); n++) {
            int i = sink.meshNode(n);
            if(nwght[j][i] > 0) out += nwght[j][i];
            else in += nwght[j][i];
        }

        Exchanger::util::gatherSum(comm, in);
        Exchanger::util::gatherSum(comm, out);

        journal::info_t info("CitcomS-AreaWeightedNormal-area");
        info << journal::at(__HERE__)
             << "Partial surface area: " << j << " "
             << in << " " << out << journal::endl;
    }
    */

    return total_area;
}


double AreaWeightedNormal::computeOutflow(const Velo& V,
					  const MPI_Comm& comm,
					  const Sink& sink) const
{
    /* integrate dot(V,n) over the surface  */
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
