// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <portinfo>
#include <cmath>
#include <iostream>
#include "Array2D.h"
#include "Boundary.h"
#include "Mapping.h"
#include "global_defs.h"
#include "AreaWeightedNormal.h"



AreaWeightedNormal::AreaWeightedNormal(const Boundary* boundary,
				       const All_variables* E,
				       const FineGridMapping* fgmapping) :
    size_(boundary->size()),
    toleranceOutflow_(E->control.tole_comp),
    nwght(new double[size_ * dim_])
{
    for(int i=0; i<size_*dim_; i++)
	nwght[i] = 0;

    computeWeightedNormal(boundary, E, fgmapping);
}


AreaWeightedNormal::~AreaWeightedNormal() {
    delete [] nwght;
}


void AreaWeightedNormal::imposeConstraint(Velo& V) const {

    double outflow = computeOutflow(V);
    std::cout << " Net outflow before boundary velocity correction " << outflow << std::endl;

    if (fabs(outflow) > toleranceOutflow_) {
	reduceOutflow(V, outflow);

	outflow = computeOutflow(V);
	std::cout << " Net outflow after boundary velocity correction (SHOULD BE ZERO !) " << outflow << std::endl;
    }
}


void AreaWeightedNormal::computeWeightedNormal(const Boundary* boundary,
					       const All_variables* E,
					       const FineGridMapping* fgmapping) {
    const int nodesPerElement = (int) pow(2, dim_);
    const int facenodes[]={0, 1, 5, 4,
                           2, 3, 7, 6,
                           1, 2, 6, 5,
                           0, 4, 7, 3,
                           4, 5, 6, 7,
                           0, 3, 2, 1};

    int nodest = nodesPerElement * E->lmesh.nel;
    int* bnodes = new int[nodest];
    for(int i=0; i<nodest; i++) bnodes[i] = -1;

    // Assignment of the local boundary node numbers
    // to bnodes elements array
    for(int n=0; n<E->lmesh.nel; n++) {
	for(int j=0; j<nodesPerElement; j++) {
	    int gnode = E->IEN[E->mesh.levmax][1][n+1].node[j+1];
	    for(int k=0; k<fgmapping->size(); k++) {
		if(gnode == fgmapping->bid2gid(k)) {
		    bnodes[n*nodesPerElement+j] = k;
		    break;
		}
	    }
	}
    }

    double garea[dim_][2];
    for(int i=0; i<dim_; i++)
	for(int j=0; j<2; j++)
	    garea[i][j] = 0.0;

    for(int n=0; n<E->lmesh.nel; n++) {
	// Loop over element faces
	for(int i=0; i<6; i++) {
	    // Checking of diagonal nodal faces
	    if((bnodes[n*nodesPerElement+facenodes[i*4]] >=0) &&
	       (bnodes[n*nodesPerElement+facenodes[i*4+1]] >=0) &&
	       (bnodes[n*nodesPerElement+facenodes[i*4+2]] >=0) &&
	       (bnodes[n*nodesPerElement+facenodes[i*4+3]] >=0)) {

		double xc[12], normal[dim_];
		for(int j=0; j<4; j++) {
		    int lnode = bnodes[n*nodesPerElement+facenodes[i*4+j]];
		    if(lnode >= boundary->size())
			std::cout <<" lnode = " << lnode
				  << " size " << boundary->size()
				  << std::endl;
		    for(int l=0; l<dim_; l++)
			xc[j*dim_+l] = boundary->X(l,lnode);
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

		for(int l=0; l<dim_; l++)
		    normal[l] /= area;

		if(xc[0] == xc[6])
		    area = fabs(0.5 * (xc[2]+xc[8]) * (xc[8]-xc[2])
				* (xc[7]-xc[1]) * sin(xc[0]));
		if(xc[1] == xc[7])
		    area = fabs(0.5 * (xc[2]+xc[8]) * (xc[8]-xc[2])
				* (xc[6]-xc[0]));
		if(xc[2] == xc[8])
		    area = fabs(xc[2] * xc[8] * (xc[7]-xc[1])
				* (xc[6]-xc[0]) * sin(0.5*(xc[0]+xc[6])));

		for(int l=0; l<dim_; l++) {
		    if(normal[l] > 0.999 ) garea[l][0] += area;
		    if(normal[l] < -0.999 ) garea[l][1] += area;
		}
		for(int j=0; j<4; j++) {
		    int lnode = bnodes[n*nodesPerElement+facenodes[i*4+j]];
		    for(int l=0; l<dim_; l++)
			nwght[lnode*dim_+l] += normal[l] * area/4.;
		}
	    } // end of check of nodes
	} // end of loop over faces
    } // end of loop over elements
    delete [] bnodes;
}


double AreaWeightedNormal::computeOutflow(const Velo& V) const {
    double outflow = 0;
    for(int n=0; n<V.size(); n++)
	for(int j=0; j<dim_; j++)
	    outflow += V[j][n] * nwght[n*dim_+j];

    return outflow;
}


void AreaWeightedNormal::reduceOutflow(Velo& V, const double outflow) const {
    double total_area = 0;
    for(int n=0; n<size_; n++)
	for(int j=0; j<dim_; j++)
	    total_area += fabs(nwght[n*3+j]);

    for(int n=0; n<size_; n++) {
	for(int j=0; j<dim_; j++)
	    if(fabs(nwght[n*dim_+j]) > 1.e-10) {
		V[j][n] -= outflow * nwght[n*dim_+j]
		    / (total_area * fabs(nwght[n*dim_+j]));
	    }
    }
}



// version
// $Id: AreaWeightedNormal.cc,v 1.2 2003/10/22 20:33:51 ces74 Exp $

// End of file
