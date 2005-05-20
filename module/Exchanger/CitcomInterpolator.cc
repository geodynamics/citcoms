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
#include <stdexcept>
#include <cmath>
#include "global_bbox.h"
#include "global_defs.h"
#include "journal/journal.h"
#include "Exchanger/BoundedBox.h"
#include "Exchanger/BoundedMesh.h"
#include "CitcomInterpolator.h"

using Exchanger::Array2D;
using Exchanger::BoundedBox;
using Exchanger::BoundedMesh;
using Exchanger::DIM;
using Exchanger::NODES_PER_ELEMENT;
using Exchanger::STRESS_DIM;


CitcomInterpolator::CitcomInterpolator(const BoundedMesh& boundedMesh,
				       Array2D<int,1>& meshNode,
				       const All_variables* e) :
    Exchanger::Interpolator(),
    E(e)
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    init(boundedMesh, meshNode);
    selfTest(boundedMesh, meshNode);

    elem_.print("CitcomS-CitcomInterpolator-elem");
    shape_.print("CitcomS-CitcomInterpolator-shape");
}


CitcomInterpolator::~CitcomInterpolator()
{}


void CitcomInterpolator::interpolatePressure(Array2D<double,1>& P)
{
    P.assign(size(), 0);

    const int mm = 1;
    for(int i=0; i<size(); i++) {
        int n1 = elem_[0][i];
        for(int k=0; k<NODES_PER_ELEMENT; k++) {
            int node = E->ien[mm][n1].node[k+1];
            P[0][i] += shape_[k][i] * E->P[mm][node];
        }
    }
}


void CitcomInterpolator::interpolateStress(Array2D<double,STRESS_DIM>& S)
{
    S.assign(size(), 0);

    const int mm = 1;
    for(int i=0; i<size(); i++) {
        int n1 = elem_[0][i];
        for(int k=0; k<NODES_PER_ELEMENT; k++) {
            int node = E->ien[mm][n1].node[k+1] - 1;
            for(int d=0; d<STRESS_DIM; d++)
                S[d][i] += shape_[k][i] * E->gstress[mm][node*STRESS_DIM+d+1];
        }
    }
}


void CitcomInterpolator::interpolateTemperature(Array2D<double,1>& T)
{
    T.assign(size(), 0);

    const int mm = 1;
    for(int i=0; i<size(); i++) {
        int n1 = elem_[0][i];
        for(int k=0; k<NODES_PER_ELEMENT; k++) {
            int node = E->ien[mm][n1].node[k+1];
            T[0][i] += shape_[k][i] * E->T[mm][node];
        }
    }
}


void CitcomInterpolator::interpolateVelocity(Array2D<double,DIM>& V)
{
    V.assign(size(), 0);

    const int mm = 1;
    for(int i=0; i<size(); i++) {
        int n1 = elem_[0][i];
        for(int k=0; k<NODES_PER_ELEMENT; k++) {
            int node = E->ien[mm][n1].node[k+1];
            for(int d=0; d<DIM; d++)
                V[d][i] += shape_[k][i] * E->sphere.cap[mm].V[d+1][node];
        }
    }
}


// private functions

void CitcomInterpolator::init(const BoundedMesh& boundedMesh,
			      Array2D<int,1>& meshNode)
{
    journal::debug_t debug("CitcomS-Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    const int mm = 1;

    elem_.reserve(boundedMesh.size());
    shape_.reserve(boundedMesh.size());

    Array2D<double,DIM*DIM> etaAxes;     // axes of eta coord.
    Array2D<double,DIM> inv_length_sq;   // reciprocal of (length of etaAxes)^2
    computeElementGeometry(etaAxes, inv_length_sq);

    // get remote BoundedBox
    BoundedBox remoteBBox(boundedMesh.tightBBox());

    // get local BoundedBox
    BoundedBox bbox(DIM);
    if(E->parallel.nprocxy == 12) {
	// for CitcomS Full
	fullGlobalBoundedBox(bbox, E);
    }
    else {
	// for CitcomS Regional
	regionalGlobalBoundedBox(bbox, E);
    }


    // z is the range of depth in current processor
    std::vector<double> z(E->lmesh.noz);
    for(size_t i=0; i<z.size(); ++i)
	z[i] = E->sx[mm][DIM][i+1];

    for(int n=0; n<boundedMesh.size(); ++n) {

	std::vector<double> x(DIM);
	for(int d=0; d<DIM; ++d)
	    x[d] = boundedMesh.X(d,n);

	// sometimes after coordinate conversion, surface nodes of different
	// solvers won't line up, need this special treatment --
	// if x is a little bit above my top surface, move it back to surface
	if(x[DIM-1] == remoteBBox[1][DIM-1]) {
	    double offtop = x[DIM-1]/bbox[1][DIM-1] - 1.0;
	    if(offtop < 1e-5 && offtop > 0)
		x[DIM-1] = bbox[1][DIM-1];
	}

	// skip if x is not inside bbox
	if(!isInside(x, bbox)) continue;

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


void CitcomInterpolator::computeElementGeometry(Array2D<double,DIM*DIM>& etaAxes,
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

    etaAxes.print("CitcomS-CitcomInterpolator-etaAxes");
    inv_length_sq.print("CitcomS-CitcomInterpolator-inv-length-sq");
}


int CitcomInterpolator::bisectInsertPoint(double x,
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


bool CitcomInterpolator::elementInverseMapping(std::vector<double>& elmShape,
					       const std::vector<double>& x,
					       const Array2D<double,DIM*DIM>& etaAxes,
					       const Array2D<double,DIM>& inv_length_sq,
					       int element,
					       double accuracy)
{
    const int mm = 1;
    bool found = false;
    int count = 0;
    std::vector<double> eta(DIM); // initial eta = (0,0,0)

    while (1) {
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
		eta[d] += 0.9 * deta[d];

	// if x is inside this element, -1 < eta[d] < 1, d = 0 ... DIM
	bool outside = false;
	for(int d=0; d<DIM; ++d)
	    outside = outside || (std::abs(eta[d]) > 1.5);

	// iterate at least twice
	found = (distancesq < accuracy) && (count > 0);

	if (outside || found || (count > 100))
	    break;

	++count;

	/* Only need to iterate if this is marginal. If eta > distortion of
	   an individual element then almost certainly x is in a
	   different element ... or the mesh is terrible !  */

    }

    return found;
}


void CitcomInterpolator::getShapeFunction(std::vector<double>& shape,
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



void CitcomInterpolator::selfTest(const BoundedMesh& boundedMesh,
				  const Array2D<int,1>& meshNode) const
{
    journal::debug_t debug("CitcomS-Exchanger");
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

	    journal::firewall_t firewall("CitcomS-CitcomInterpolator");
	    firewall << journal::loc(__HERE__)
		     << "node #" << i << " tshape = " << tshape
		     << journal::newline
	    	     << xi[0] << " " << xt[0] << " "
		     << xi[1] << " " << xt[1] << " "
		     << xi[2] << " " << xt[2] << " "
		     << " norm = " << norm << journal::newline
		     << "elem interpolation functions are wrong"
		     << journal::end;
	    throw std::domain_error("CitcomInterpolator");
	}
    }
}


// version
// $Id: CitcomInterpolator.cc,v 1.3 2005/05/19 22:39:18 leif Exp $

// End of file
