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
#include "global_defs.h"
#include "journal/journal.h"
#include "BoundedBox.h"
#include "BoundedMesh.h"
#include "Interpolator.h"


Interpolator::Interpolator(const BoundedMesh& boundedMesh,
			   const All_variables* E,
			   Array2D<int,1>& meshNode)
{
    init(boundedMesh, E, meshNode);
    selfTest(boundedMesh, E, meshNode);

    elem_.print("elem");
    //shape_.print("shape");
}


void Interpolator::interpolateV(Array2D<double,DIM>& target,
				const All_variables* E) const
{
    target.assign(size(), 0);

    const int mm = 1;
    for(int i=0; i<size(); i++) {
	int n1 = elem_[0][i];
	for(int k=0; k<NODES_PER_ELEMENT; k++) {
	    int node = E->ien[mm][n1].node[k+1];
	    for(int d=0; d<DIM; d++)
		target[d][i] += shape_[k][i] * E->sphere.cap[mm].V[d+1][node];
	}
    }
}


void Interpolator::interpolateT(Array2D<double,1>& target,
				const All_variables* E) const
{
    target.assign(size(), 0);

    const int mm = 1;
    for(int i=0; i<size(); i++) {
	int n1 = elem_[0][i];
	for(int k=0; k<NODES_PER_ELEMENT; k++) {
	    int node = E->ien[mm][n1].node[k+1];
	    target[0][i] += shape_[k][i] * E->T[mm][node];
	}
    }
}


// private functions


// vertices of five sub-tetrahedra
const int nsub[] = {0, 2, 3, 7,
		    0, 1, 2, 5,
		    4, 7, 5, 0,
		    5, 7, 6, 2,
		    5, 7, 2, 0};


void Interpolator::init(const BoundedMesh& boundedMesh,
			const All_variables* E,
			Array2D<int,1>& meshNode)
{
    double xt[DIM], xc[DIM*NODES_PER_ELEMENT], x1[DIM], x2[DIM], x3[DIM], x4[DIM];

    findMaxGridSpacing(E);

    elem_.reserve(boundedMesh.size());
    shape_.reserve(boundedMesh.size());

    const int mm = 1;
    for(int i=0; i<boundedMesh.size(); i++) {
	for(int j=0; j<DIM; j++) xt[j] = boundedMesh.X(j,i);
        bool found = false;

	for(int n=0; n<E->lmesh.nel; n++) {

	    for(int j=0; j<NODES_PER_ELEMENT; j++) {
		int gnode = E->ien[mm][n+1].node[j+1];
		for(int k=0; k<DIM; k++) {
		    xc[j*DIM+k] = E->sx[mm][k+1][gnode];
		}
	    }

	    if(!isCandidate(xc, boundedMesh.bbox()))continue;

	    // loop over 5 sub tets in a brick element
	    for(int k=0; k<5; k++) {

		for(int m=0; m<DIM; m++) {
		    x1[m] = xc[nsub[k*4]*DIM+m];
		    x2[m] = xc[nsub[k*4+1]*DIM+m];
		    x3[m] = xc[nsub[k*4+2]*DIM+m];
		    x4[m] = xc[nsub[k*4+3]*DIM+m];
		}

		double dett, det[4];
		dett = TetrahedronVolume(x1,x2,x3,x4);
		det[0] = TetrahedronVolume(x2,x4,x3,xt);
		det[1] = TetrahedronVolume(x3,x4,x1,xt);
		det[2] = TetrahedronVolume(x1,x4,x2,xt);
		det[3] = TetrahedronVolume(x1,x2,x3,xt);

		if(dett < 0) {
		    journal::firewall_t firewall("Interpolator");
		    firewall << journal::loc(__HERE__)
			     << "Determinant evaluation is wrong"
			     << journal::newline
			     << " node " << i
			     << " " << xt[0]
			     << " " << xt[1]
			     << " " << xt[2]
			     << journal::newline;
		    for(int j=0; j<NODES_PER_ELEMENT; j++)
			firewall << xc[j*DIM]
				 << " " << xc[j*DIM+1]
				 << " " << xc[j*DIM+2]
				 << journal::newline;

		    firewall << journal::end;
		}

		// found if all det are greated than zero
		found = (det[0] > -1.e-10 &&
			 det[1] > -1.e-10 &&
			 det[2] > -1.e-10 &&
			 det[3] > -1.e-10);

		if (found) {
		    meshNode.push_back(i);
		    appendFoundElement(n, k, det, dett);
		    break;
		}
	    }
	    if(found) break;
	}
    }
    elem_.shrink();
    shape_.shrink();
}


void Interpolator::findMaxGridSpacing(const All_variables* E)
{

    if(E->parallel.nprocxy == 12) {
	// for CitcomSFull
	const double pi = 4*atan(1);
	const double cap_side = 0.5*pi / sqrt(2);  // side length of a spherical cap
	double elem_side = cap_side / E->mesh.elx;
	theta_tol = fi_tol = elem_side;
    }
    else {
	theta_tol = fi_tol = 0;
	const int m = 1;
	for(int n=0; n<E->lmesh.nel; n++) {
	    int gnode1 = E->ien[m][n+1].node[1];
	    int gnode2 = E->ien[m][n+1].node[2];
	    int gnode4 = E->ien[m][n+1].node[4];
	    theta_tol = std::max(theta_tol,
				 std::abs(E->sx[m][1][gnode2]
					  - E->sx[m][1][gnode1]));
	    fi_tol = std::max(fi_tol,
			      std::abs(E->sx[m][2][gnode4]
				       - E->sx[m][2][gnode1]));
	}
    }

    r_tol = 0;
    const int m = 1;
    for(int n=0; n<E->lmesh.nel; n++) {
	int gnode1 = E->ien[m][n+1].node[1];
	int gnode5 = E->ien[m][n+1].node[5];
	r_tol = std::max(r_tol,
			 std::abs(E->sx[m][3][gnode5]
				  - E->sx[m][3][gnode1]));
    }

    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "max grid spacing: "
	  << theta_tol << " "
	  << fi_tol << " "
	  << r_tol << journal::end;
}


bool Interpolator::isCandidate(const double* xc,
			       const BoundedBox& bbox) const
{
    std::vector<double> x(DIM);
    for(int j=0; j<NODES_PER_ELEMENT; j++) {
	for(int k=0; k<DIM; k++)
	    x[k] = xc[j*DIM+k];

	if(isInside(x, bbox)) return true;
    }
    return false;
}


double Interpolator::TetrahedronVolume(double *x1, double *x2,
				       double *x3, double *x4)  const
{
    double vol;
    //    xx[0] = x2;  xx[1] = x3;  xx[2] = x4;
    vol = det3_sub(x2,x3,x4);
    //    xx[0] = x1;  xx[1] = x3;  xx[2] = x4;
    vol -= det3_sub(x1,x3,x4);
    //    xx[0] = x1;  xx[1] = x2;  xx[2] = x4;
    vol += det3_sub(x1,x2,x4);
    //    xx[0] = x1;  xx[1] = x2;  xx[2] = x3;
    vol -= det3_sub(x1,x2,x3);
    vol /= 6.;
    return vol;
}


double Interpolator::det3_sub(double *x1, double *x2, double *x3) const
{
    return (x1[0]*(x2[1]*x3[2]-x3[1]*x2[2])
            -x1[1]*(x2[0]*x3[2]-x3[0]*x2[2])
            +x1[2]*(x2[0]*x3[1]-x3[0]*x2[1]));
}


void Interpolator::appendFoundElement(int el, int ntetra,
				      const double* det, double dett)
{
    std::vector<double> tmp(NODES_PER_ELEMENT, 0);
    tmp[nsub[ntetra*4]] = det[0]/dett;
    tmp[nsub[ntetra*4+1]] = det[1]/dett;
    tmp[nsub[ntetra*4+2]] = det[2]/dett;
    tmp[nsub[ntetra*4+3]] = det[3]/dett;

    shape_.push_back(tmp);
    elem_.push_back(el+1);
}


void Interpolator::selfTest(const BoundedMesh& boundedMesh,
			    const All_variables* E,
			    const Array2D<int,1>& meshNode) const
{
    double xc[DIM*NODES_PER_ELEMENT], xi[DIM], xt[DIM];

    for(int i=0; i<size(); i++) {
        for(int j=0; j<DIM; j++) xt[j] = boundedMesh.X(j, meshNode[0][i]);

        int n1 = elem_[0][i];

        for(int j=0; j<NODES_PER_ELEMENT; j++) {
            for(int k=0; k<DIM; k++) {
                xc[j*DIM+k] = E->sx[1][k+1][E->ien[1][n1].node[j+1]];
            }
        }

        for(int k=0; k<DIM; k++) xi[k] = 0.0;
        for(int k=0; k<DIM; k++)
            for(int j=0; j<NODES_PER_ELEMENT; j++) {
                xi[k] += xc[j*DIM+k] * shape_[j][i];
            }

        double norm = 0.0;
        for(int k=0; k<DIM; k++) norm += (xt[k]-xi[k]) * (xt[k]-xi[k]);
        if(norm > 1.e-10) {
            double tshape = 0.0;
            for(int j=0; j<NODES_PER_ELEMENT; j++)
		tshape += shape_[j][i];

	    journal::firewall_t firewall("Interpolator");
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
// $Id: Interpolator.cc,v 1.2 2003/11/11 19:29:27 tan2 Exp $

// End of file
