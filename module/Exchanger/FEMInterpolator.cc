// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <algorithm>
#include <iostream>
#include "global_defs.h"
#include "journal/journal.h"
#include "BoundedBox.h"
#include "BoundedMesh.h"
#include "FEMInterpolator.h"


FEMInterpolator::FEMInterpolator(const BoundedMesh& boundedMesh,
				 const All_variables* e,
				 Array2D<int,1>& meshNode) :
    E(e),
    elem_(0),
    shape_(0)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    init(boundedMesh, meshNode);
    selfTest(boundedMesh, meshNode);

    elem_.print("elem");
    shape_.print("shape");
}


// private functions

void FEMInterpolator::init(const BoundedMesh& boundedMesh,
			   Array2D<int,1>& meshNode)
{
    const int mm = 1;

    elem_.reserve(boundedMesh.size());
    shape_.reserve(boundedMesh.size());

    Array2D<double,DIM*DIM> etaAxes;     // axes of eta coord.
    Array2D<double,DIM> inv_length_sq;   // reciprocal of (length of etaAxes)^2
    computeElementGeometry(etaAxes, inv_length_sq);

    // z is the range of depth in current processor
    std::vector<double> z(E->lmesh.noz);
    for(size_t i=0; i<z.size(); ++i)
	z[i] = E->sx[mm][DIM][i+1];

    for(int n=0; n<boundedMesh.size(); ++n) {

	std::vector<double> x(DIM);
	for(int d=0; d<DIM; ++d)
	    x[d] = boundedMesh.X(d,n);

	// skip if x is not inside bbox
	if(!isInside(x, boundedMesh.bbox())) continue;

#if 1
	// skip if x is not in the range of depth
	if(x.back() < z.front() || x.back() > z.back()) continue;
	int elz = bisectInsertPoint(x.back(), z);
	// Since the mesh of CitcomS is structural and regular
	// we only need to loop over elements in a constant depth
	for(int el=elz; el<E->lmesh.nel; el+=E->lmesh.elz) {
#else
	// If the mesh is not regular, loop over all elements
	for(int el=0; el<E->lmesh.nel; ++el) {
#endif

	    std::vector<double> elmShape(NODES_PER_ELEMENT);
	    double accuracy = E->control.accuracy * E->control.accuracy
 		              * E->eco[mm][el+1].area;
	    bool found = elementInverseMapping(elmShape, x,
					       etaAxes, inv_length_sq,
					       el, accuracy);

	    if(found) {
		meshNode.push_back(n);
		elem_.push_back(el+1);
		shape_.push_back(elmShape);
		break;
	    }
	}
    }
}


void FEMInterpolator::computeElementGeometry(Array2D<double,DIM*DIM>& etaAxes,
					     Array2D<double,DIM>& inv_length_sq) const
{
    etaAxes.resize(E->lmesh.nel);
    inv_length_sq.resize(E->lmesh.nel);

    const int mm = 1;
    const int surfnodes = 4;  // # of nodes on element's face

    // node 1, 2, 5, 6 are in the face that is on positive x axis
    // node 2, 3, 6, 7 are in the face that is on positive y axis
    // node 4, 5, 6, 7 are in the face that is on positive z axis
    // node 0, 3, 4, 7 are in the face that is on negative x axis
    // node 0, 1, 4, 5 are in the face that is on negative y axis
    // node 0, 1, 2, 3 are in the face that is on negative z axis
    // see comment of getShapeFunction() for the ordering of nodes
    const int high[] = {1, 2, 5, 6,
  			  2, 3, 6, 7,
			  4, 5, 6, 7};
    const int low[] = {0, 3, 4, 7,
		         0, 1, 4, 5,
		         0, 1, 2, 3};

    std::vector<int> node(NODES_PER_ELEMENT);

    for(int element=0; element<E->lmesh.nel; ++element) {

	for(int k=0; k<NODES_PER_ELEMENT; ++k)
	    node[k] = E->ien[mm][element+1].node[k+1];

	for(int n=0; n<DIM; ++n) {

	    int k = n * surfnodes;
	    for(int d=0; d<DIM; ++d) {
		double lowmean = 0;
		double highmean = 0;
		for(int i=0; i<surfnodes; ++i) {
		    highmean += E->sx[mm][d+1][node[high[k+i]]];
		    lowmean += E->sx[mm][d+1][node[low[k+i]]];
		}

		etaAxes[n*DIM+d][element] = 0.5 * (highmean - lowmean)
		                          / surfnodes;
	    }
	}
    }

    for(int element=0; element<E->lmesh.nel; ++element)
	for(int n=0; n<DIM; ++n) {
	    double lengthsq = 0;
	    for(int d=0; d<DIM; ++d)
		lengthsq += etaAxes[n*DIM+d][element]
		          * etaAxes[n*DIM+d][element];

	    inv_length_sq[n][element] = 1 / lengthsq;
	}

    etaAxes.print("etaAxes");
    inv_length_sq.print("inv-length-sq");
}


int FEMInterpolator::bisectInsertPoint(double x,
				const std::vector<double>& v) const
{
    int low = 0;
    int high = v.size();
    int insert_point = (low + high) / 2;

    while(low < high) {
	if(x < v[insert_point])
	    high = insert_point;
	else if(x > v[insert_point+1])
	    low = insert_point;
	else
	    break;

	insert_point = (low + high) / 2;
    }

    return insert_point;
}


bool FEMInterpolator::elementInverseMapping(std::vector<double>& elmShape,
				    const std::vector<double>& x,
				    const Array2D<double,DIM*DIM>& etaAxes,
				    const Array2D<double,DIM>& inv_length_sq,
				    int element,
				    double accuracy)
{
    const int mm = 1;
    bool found = false;
    bool keep_going = true;
    int count = 0;
    std::vector<double> eta(DIM); // initial eta = (0,0,0)

    do {
	getShapeFunction(elmShape, eta);

	std::vector<double> xx(DIM);
	for(int k=0; k<NODES_PER_ELEMENT; ++k) {
	    int node = E->ien[mm][element+1].node[k+1];
	    for(int d=0; d<DIM; ++d)
		xx[d] += E->sx[mm][d+1][node] * elmShape[k];
	}

	std::vector<double> dx(DIM);
	double distancesq = 0;
	for(int d=0; d<DIM; ++d) {
	    dx[d] = x[d] - xx[d];
	    distancesq += dx[d] * dx[d];
	}

	// correction of eta
	std::vector<double> deta(DIM);
	for(int d=0; d<DIM; ++d) {
	    for(int i=0; i<DIM; ++i)
		deta[d] += dx[i] * etaAxes[d*DIM+i][element];

	    deta[d] *= inv_length_sq[d][element];
	}

	if(count == 0)
	    for(int d=0; d<DIM; ++d)
		eta[d] += deta[d];
	else  // Damping
	    for(int d=0; d<DIM; ++d)
		eta[d] += 0.8 * deta[d];

	// if x is inside this element, -1 < eta[d] < 1, d = 0 ... DIM
	bool outside = false;
	for(int d=0; d<DIM; ++d)
	    outside = outside || (std::abs(eta[d]) > 2);

	found = distancesq < accuracy;
	keep_going = (!found) && (count < 100) && (!outside);
	++count;

	/* Only need to iterate if this is marginal. If eta > distortion of
	   an individual element then almost certainly x is in a
	   different element ... or the mesh is terrible !  */

    } while(keep_going);

    return found;
}


void FEMInterpolator::getShapeFunction(std::vector<double>& shape,
				       const std::vector<double>& eta) const
{
    // the ordering of nodes in an element
    // node #: eta coordinate
    // node 0: (-1, -1, -1)
    // node 1: ( 1, -1, -1)
    // node 2: ( 1,  1, -1)
    // node 3: (-1,  1, -1)
    // node 4: (-1, -1,  1)
    // node 5: ( 1, -1,  1)
    // node 6: ( 1,  1,  1)
    // node 7: (-1,  1,  1)

    shape[0] = 0.125 * (1.0-eta[0]) * (1.0-eta[1]) * (1.0-eta[2]);
    shape[1] = 0.125 * (1.0+eta[0]) * (1.0-eta[1]) * (1.0-eta[2]);
    shape[2] = 0.125 * (1.0+eta[0]) * (1.0+eta[1]) * (1.0-eta[2]);
    shape[3] = 0.125 * (1.0-eta[0]) * (1.0+eta[1]) * (1.0-eta[2]);
    shape[4] = 0.125 * (1.0-eta[0]) * (1.0-eta[1]) * (1.0+eta[2]);
    shape[5] = 0.125 * (1.0+eta[0]) * (1.0-eta[1]) * (1.0+eta[2]);
    shape[6] = 0.125 * (1.0+eta[0]) * (1.0+eta[1]) * (1.0+eta[2]);
    shape[7] = 0.125 * (1.0-eta[0]) * (1.0+eta[1]) * (1.0+eta[2]);
}



void FEMInterpolator::selfTest(const BoundedMesh& boundedMesh,
			       const Array2D<int,1>& meshNode) const
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    for(int i=0; i<size(); i++) {

	// xt is the node that we like to interpolate to
        std::vector<double> xt(DIM);
        for(int j=0; j<DIM; j++)
	    xt[j] = boundedMesh.X(j, meshNode[0][i]);

	// xi is the result of interpolation
        std::vector<double> xi(DIM);
        int n1 = elem_[0][i];
	for(int j=0; j<NODES_PER_ELEMENT; j++) {
	    int node = E->ien[1][n1].node[j+1];
	    for(int k=0; k<DIM; k++)
                xi[k] += E->sx[1][k+1][node] * shape_[j][i];
	}

	// if xi and xt are not coincide, the interpolation is wrong
        double norm = 0.0;
        for(int k=0; k<DIM; k++)
	    norm += (xt[k]-xi[k]) * (xt[k]-xi[k]);

        if(norm > 1.e-10) {
            double tshape = 0.0;
            for(int j=0; j<NODES_PER_ELEMENT; j++)
		tshape += shape_[j][i];

	    journal::firewall_t firewall("FEMInterpolator");
            firewall << journal::loc(__HERE__)
		     << "node #" << i << " tshape = " << tshape
		     << journal::newline
		     << xi[0] << " " << xt[0] << " "
		     << xi[1] << " " << xt[1] << " "
		     << xi[2] << " " << xt[2] << " "
		     << " norm = " << norm << journal::newline
		     << "elem interpolation functions are wrong"
		     << journal::end;
        }
    }
}


// version
// $Id: FEMInterpolator.cc,v 1.7 2004/02/05 19:45:09 tan2 Exp $

// End of file
