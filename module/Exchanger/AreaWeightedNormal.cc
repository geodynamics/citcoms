// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <cmath>
#include "global_defs.h"
#include "journal/journal.h"
#include "utilTemplate.h"
#include "Array2D.h"
#include "Boundary.h"
#include "Sink.h"
#include "AreaWeightedNormal.h"



AreaWeightedNormal::AreaWeightedNormal(const MPI_Comm& comm,
				       const Boundary& boundary,
				       const Sink& sink,
				       const All_variables* E) :
    size_(boundary.size()),
    toleranceOutflow_(E->control.tole_comp),
    nwght(size_ * DIM)
{
    computeWeightedNormal(boundary, sink, E);
    computeTotalArea(comm, sink);
}


void AreaWeightedNormal::imposeConstraint(Velo& V,
					  const MPI_Comm& comm,
					  const Sink& sink) const
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in AreaWeightedNormal::imposeConstraint" << journal::end;

    double outflow = computeOutflow(V, comm, sink);
    debug << journal::loc(__HERE__)
	  << "Net outflow before boundary velocity correction "
	  << outflow << journal::end;

    if (std::abs(outflow) > toleranceOutflow_) {
	reduceOutflow(V, outflow, sink);

	outflow = computeOutflow(V, comm, sink);
	debug << journal::loc(__HERE__)
	      << "Net outflow after boundary velocity correction (SHOULD BE ZERO !) "
	      << outflow << journal::end;
    }
}


void AreaWeightedNormal::computeWeightedNormal(const Boundary& boundary,
					       const Sink& sink,
					       const All_variables* E)
{
    const int facenodes[]={0, 1, 5, 4,
                           2, 3, 7, 6,
                           1, 2, 6, 5,
                           0, 4, 7, 3,
                           4, 5, 6, 7,
                           0, 3, 2, 1};

    const int nodest = NODES_PER_ELEMENT * E->lmesh.nel;
    std::vector<int> bnodes(nodest, -1);

    // Assignment of the local boundary node numbers
    // to bnodes elements array
    for(int n=0; n<E->lmesh.nel; n++) {
	for(int j=0; j<NODES_PER_ELEMENT; j++) {
	    int gnode = E->ien[1][n+1].node[j+1];
	    for(int k=0; k<boundary.size(); k++) {
		if(gnode == boundary.nodeID(k)) {
		    bnodes[n*NODES_PER_ELEMENT+j] = k;
		    break;
		}
	    }
	}
    }

    double garea[DIM][2];
    for(int i=0; i<DIM; i++)
	for(int j=0; j<2; j++)
	    garea[i][j] = 0.0;

    for(int n=0; n<E->lmesh.nel; n++) {
	// Loop over element faces
	for(int i=0; i<6; i++) {
	    // Checking of diagonal nodal faces
	    if((bnodes[n*NODES_PER_ELEMENT+facenodes[i*4]] >= 0) &&
	       (bnodes[n*NODES_PER_ELEMENT+facenodes[i*4+1]] >= 0) &&
	       (bnodes[n*NODES_PER_ELEMENT+facenodes[i*4+2]] >= 0) &&
	       (bnodes[n*NODES_PER_ELEMENT+facenodes[i*4+3]] >= 0)) {

		double xc[4*DIM], normal[DIM];
		for(int j=0; j<4; j++) {
		    int lnode = bnodes[n*NODES_PER_ELEMENT+facenodes[i*4+j]];
		    for(int l=0; l<DIM; l++)
			xc[j*DIM+l] = boundary.X(l,lnode);
		}

		normal[0] = (xc[4]-xc[1])*(xc[11]-xc[2])
		          - (xc[5]-xc[2])*(xc[10]-xc[1]);
		normal[1] = (xc[5]-xc[2])*(xc[9]-xc[0])
			  - (xc[3]-xc[0])*(xc[11]-xc[2]);
		normal[2] = (xc[3]-xc[0])*(xc[10]-xc[1])
			  - (xc[4]-xc[1])*(xc[9]-xc[0]);
		double area = sqrt(normal[0]*normal[0]
				   + normal[1]*normal[1]
				   + normal[2]*normal[2]);

		for(int l=0; l<DIM; l++)
		    normal[l] /= area;

		if(xc[0] == xc[6])
		    area = std::abs(0.5 * (xc[2]+xc[8]) * (xc[8]-xc[2])
				    * (xc[7]-xc[1]) * sin(xc[0]));
		if(xc[1] == xc[7])
		    area = std::abs(0.5 * (xc[2]+xc[8]) * (xc[8]-xc[2])
				    * (xc[6]-xc[0]));
		if(xc[2] == xc[8])
		    area = std::abs(xc[2] * xc[8] * (xc[7]-xc[1])
				    * (xc[6]-xc[0]) * sin(0.5*(xc[0]+xc[6])));

		for(int l=0; l<DIM; l++) {
		    if(normal[l] > 0.999 ) garea[l][0] += area;
		    if(normal[l] < -0.999 ) garea[l][1] += area;
		}
		for(int j=0; j<4; j++) {
		    int lnode = bnodes[n*NODES_PER_ELEMENT+facenodes[i*4+j]];
		    for(int l=0; l<DIM; l++)
			nwght[lnode*DIM+l] += normal[l] * area/4.;
		}
	    } // end of check of nodes
	} // end of loop over faces
    } // end of loop over elements
}


void AreaWeightedNormal::computeTotalArea(const MPI_Comm& comm,
					  const Sink& sink)
{
    total_area_ = 0;
    for(int n=0; n<sink.size(); n++) {
	int i = sink.meshNode(n);
	for(int j=0; j<DIM; j++)
	    total_area_ += std::abs(nwght[i*DIM+j]);
    }

    util::gatherSum(comm, total_area_);
}


double AreaWeightedNormal::computeOutflow(const Velo& V,
					  const MPI_Comm& comm,
					  const Sink& sink) const
{
    double outflow = 0;
    for(int n=0; n<sink.size(); n++) {
	int i = sink.meshNode(n);
	for(int j=0; j<DIM; j++)
	    outflow += V[j][n] * nwght[i*DIM+j];
    }

    util::gatherSum(comm, outflow);

    return outflow;
}


void AreaWeightedNormal::reduceOutflow(Velo& V, double outflow,
				       const Sink& sink) const
{
    for(int n=0; n<sink.size(); n++) {
	int i = sink.meshNode(n);
	for(int j=0; j<DIM; j++)
	    if(std::abs(nwght[i*DIM+j]) > 1.e-10) {
		V[j][n] -= outflow * nwght[i*DIM+j]
		    / (total_area_ * std::abs(nwght[i*DIM+j]));
	    }
    }
}



// version
// $Id: AreaWeightedNormal.cc,v 1.7 2003/11/21 23:15:12 tan2 Exp $

// End of file
