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
    shape_.print("shape");
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


#if 0
/* ===========================================================================
   Function to create the element locations from  the node positions.
   ===========================================================================	*/

void element_locations_from_nodes(struct All_variables *E)
{

    for(int element=1; element<=E->lmesh.nel; element++) {

	int n1=E->ien[1][element].node[1];
	int n2=E->ien[1][element].node[2];
	int n3=E->ien[1][element].node[3];
	int n4=E->ien[1][element].node[4];
	int n5=E->ien[1][element].node[5];
	int n6=E->ien[1][element].node[6];
	int n7=E->ien[1][element].node[7];
	int n8=E->ien[1][element].node[8];


	/* 1: x direction */

	double xlowmean = (E->x[1][1][n1] +
			   E->x[1][1][n2] +
			   E->x[1][1][n5] +
			   E->x[1][1][n6] )  * 0.25;
	double xhighmean = (E->x[1][1][n3] +
			    E->x[1][1][n4] +
			    E->x[1][1][n7] +
			    E->x[1][1][n8] )  * 0.25;
	double zlowmean = (E->x[1][2][n1] +
			   E->x[1][2][n2] +
			   E->x[1][2][n5] +
			   E->x[1][2][n6] )  * 0.25;
	double zhighmean = (E->x[1][2][n3] +
			    E->x[1][2][n4] +
			    E->x[1][2][n7] +
			    E->x[1][2][n8] )  * 0.25;
	double ylowmean = (E->x[1][3][n1] +
			   E->x[1][3][n2] +
			   E->x[1][3][n5] +
			   E->x[1][3][n6] )  * 0.25;
	double yhighmean = (E->x[1][3][n3] +
			    E->x[1][3][n4] +
			    E->x[1][3][n7] +
			    E->x[1][3][n8] )  * 0.25;

	ntl_dirns[element][1][1] = xhighmean - xlowmean;
	ntl_dirns[element][1][2] = zhighmean - zlowmean;
	ntl_dirns[element][1][3] = yhighmean - ylowmean;

	double ntl_size = sqrt(ntl_dirns[element][1][1] *
			       ntl_dirns[element][1][1] +
			       ntl_dirns[element][1][2] *
			       ntl_dirns[element][1][2] +
			       ntl_dirns[element][1][3] *
			       ntl_dirns[element][1][3] );

	ntl_recip_size[element][1] = 1.0 / ntl_size ;

	ntl_dirns[element][1][1] *= ntl_recip_size[element][1];
	ntl_dirns[element][1][2] *= ntl_recip_size[element][1];
	ntl_dirns[element][1][3] *= ntl_recip_size[element][1];

	ntl_recip_size[element][1] *= 2.0;

	/* 2: z direction */

	xlowmean =  (E->x[1][1][n1] +
		     E->x[1][1][n4] +
		     E->x[1][1][n5] +
		     E->x[1][1][n8] )  * 0.25;
	xhighmean = (E->x[1][1][n2] +
		     E->x[1][1][n3] +
		     E->x[1][1][n6] +
		     E->x[1][1][n7] )  * 0.25;
	zlowmean =  (E->x[1][2][n1] +
		     E->x[1][2][n4] +
		     E->x[1][2][n5] +
		     E->x[1][2][n8] )  * 0.25;
	zhighmean = (E->x[1][2][n2] +
		     E->x[1][2][n3] +
		     E->x[1][2][n6] +
		     E->x[1][2][n7] )  * 0.25;
	ylowmean =  (E->x[1][3][n1] +
		     E->x[1][3][n4] +
		     E->x[1][3][n5] +
		     E->x[1][3][n8] )  * 0.25;
	yhighmean = (E->x[1][3][n2] +
		     E->x[1][3][n3] +
		     E->x[1][3][n6] +
		     E->x[1][3][n7] )  * 0.25;

	ntl_dirns[element][2][1] = xhighmean - xlowmean;
	ntl_dirns[element][2][2] = zhighmean - zlowmean;
	ntl_dirns[element][2][3] = yhighmean - ylowmean;

	ntl_size = sqrt(ntl_dirns[element][2][1] *
			ntl_dirns[element][2][1] +
			ntl_dirns[element][2][2] *
			ntl_dirns[element][2][2] +
			ntl_dirns[element][2][3] *
			ntl_dirns[element][2][3] );

	ntl_recip_size[element][2] = 1.0 / ntl_size;

	ntl_dirns[element][2][1] *= ntl_recip_size[element][2];
	ntl_dirns[element][2][2] *= ntl_recip_size[element][2];
	ntl_dirns[element][2][3] *= ntl_recip_size[element][2];

	ntl_recip_size[element][2] *= 2.0;

	/* 3: y direction */

	xlowmean =  (E->x[1][1][n1] +
		     E->x[1][1][n2] +
		     E->x[1][1][n3] +
		     E->x[1][1][n4] )  * 0.25;
	xhighmean = (E->x[1][1][n5] +
		     E->x[1][1][n6] +
		     E->x[1][1][n7] +
		     E->x[1][1][n8] )  * 0.25;
	zlowmean =  (E->x[1][2][n1] +
		     E->x[1][2][n2] +
		     E->x[1][2][n3] +
		     E->x[1][2][n4] )  * 0.25;
	zhighmean = (E->x[1][2][n5] +
		     E->x[1][2][n6] +
		     E->x[1][2][n7] +
		     E->x[1][2][n8] )  * 0.25;
	ylowmean =  (E->x[1][3][n1] +
		     E->x[1][3][n2] +
		     E->x[1][3][n3] +
		     E->x[1][3][n4] )  * 0.25;
	yhighmean = (E->x[1][3][n5] +
		     E->x[1][3][n6] +
		     E->x[1][3][n7] +
		     E->x[1][3][n8] )  * 0.25;

	ntl_dirns[element][3][1] = xhighmean - xlowmean;
	ntl_dirns[element][3][2] = zhighmean - zlowmean;
	ntl_dirns[element][3][3] = yhighmean - ylowmean;

	ntl_size = sqrt(ntl_dirns[element][3][1] *
			ntl_dirns[element][3][1] +
			ntl_dirns[element][3][2] *
			ntl_dirns[element][3][2] +
			ntl_dirns[element][3][3] *
			ntl_dirns[element][3][3] );

	ntl_recip_size[element][3] = 1.0 / ntl_size ;

	ntl_dirns[element][3][1] *= ntl_recip_size[element][3];
	ntl_dirns[element][3][2] *= ntl_recip_size[element][3];
	ntl_dirns[element][3][3] *= ntl_recip_size[element][3];

	ntl_recip_size[element][3] *= 2.0;

    } // end of for loop

    /*RAA: data checking */
    /*   if(E->control.verbose)
	 for(lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)
	 for (i=1; i<=E->mesh.NEL[lev];i++)
	 fprintf(stderr,"checking element %d, lev: %d,  area: %g,    ntl_size1: %g, ntl_size2: %g, ntl_size3: %g\n",i,lev,E->ECO[lev][i].area,E->ECO[lev][i].ntl_size[1],E->ECO[lev][i].ntl_size[2],E->ECO[lev][i].ntl_size[3]);
    */
    /*
      if(E->control.verbose)
      for (i=1; i<=E->mesh.nel;i++)
      fprintf(stderr,"checking element %d,  area: %g,  size1: %g, size2: %g, size3: %g\n",i,E->eco[i].area,E->eco[i].size[1],E->eco[i].size[2],E->eco[i].size[3]);
    */

    return;
}


void get_element_coords(struct All_variables *E,
			int el,
			int num,
			standard_precision *x,
			standard_precision *z,
			standard_precision *y,
			standard_precision *eta1,
			standard_precision *eta2,
			standard_precision *eta3,
			int level
			)
{
    int k,kk;
    int node;
    int lnode[28]; /* what's the #defined variable for the max nodes/element ? */

    standard_precision xx1,xx2,xx3;
    standard_precision x1,x2,x3;
    standard_precision etadash1,etadash2,etadash3;
    standard_precision distance;
    standard_precision dirn[5][4],mag;
    standard_precision lN[ELNMAX+1];

    standard_precision area_1;

    const int dims = E->mesh.nsd;
    const int ends = enodes[dims];


    /* initial guess */

    *eta1 = *eta2 = *eta3 = 0.0;
    kk = 0;

    for(k=1;k<=ends;k++) {
	node = E->ien[1][el].node[k];
	if((E->NODE[level][node] & (PER_OFFSIDE/* | OFFSIDE*/))) { /*RAA: 1/11/01, added (..| OFFSIDE) for perx and y */
	    /* This node has ambiguous coordinates ! */

	    x1 = E->X[level][1][node] - E->ECO[level][el].centre[1];
	    x2 = E->X[level][2][node] - E->ECO[level][el].centre[2];
	    x3 = E->X[level][3][node] - E->ECO[level][el].centre[3];

	}  /*end of 'if' PER_OFFSIDE*/
	lnode[k] = node;
    }

	do {
	    shape_fn(E,el,lN,*eta1,*eta2,*eta3,level);

	    /*------------------------------------------------------*/

	    /* If periodic, we want actual (not wrapped around) coordinates
	       NB - this currently assumes only periodic in x direction -
	       and will need to be extended to y direction
	    */

	    xx1=xx2=xx3=0.0;
	    for(k=1;k<=ends;k++) {
		node = lnode[k];

		xx1 += E->X[level][1][node] * lN[k];
		xx2 += E->X[level][2][node] * lN[k];
		xx3 += E->X[level][3][node] * lN[k];
	    }

	    x1 = x[num] - xx1;
	    x2 = z[num] - xx2;
	    x3 = y[num] - xx3;

	    distance = (x1*x1+x2*x2+x3*x3);

	    etadash1 = ( x1 * ntl_dirns[el][1][1] +
			 x2 * ntl_dirns[el][1][2] +
			 x3 * ntl_dirns[el][1][3] ) * ntl_recip_size[el][1];

	    etadash2 = ( x1 * ntl_dirns[el][2][1] +
			 x2 * ntl_dirns[el][2][2] +
			 x3 * ntl_dirns[el][2][3] ) * ntl_recip_size[el][2];

	    etadash3 = ( x1 * ntl_dirns[el][3][1] +
			 x2 * ntl_dirns[el][3][2] +
			 x3 * ntl_dirns[el][3][3] ) * ntl_recip_size[el][3];

	    if(kk == 0) {
		*eta1 += etadash1;
		*eta2 += etadash2;
		*eta3 += etadash3;
	    }
	    else /* Damping */{
		*eta1 += 0.8 * etadash1;
		*eta2 += 0.8 * etadash2;
		*eta3 += 0.8 * etadash3;
	    }

	    if(++kk > 10)
		fprintf(stderr,"%d ... Tracer %d/%d in element %d ... eta (%g,%g,%g v %g,%g,%g) -> distance %g (%g,%g,%g)\n",kk,
			num,level,el,*eta1,*eta2,*eta3,x[num],z[num],y[num],distance,xx1,xx2,xx3);

	    /* Only need to iterate if this is marginal. If eta > distortion of
	       an individual element then almost certainly the tracer is in a different element ...
	       or the mesh is terrible !  */

	} while((distance > E->ECO[level][el].area * E->control.accuracy * E->control.accuracy) &&
		(fabs(*eta1) < 1.5) && (fabs(*eta2) < 1.5) && (fabs(*eta3) < 1.5) &&
		(kk < 100)); /*RAA: changed 1.5 to 5.0 in this line, to correspond with 2D case */


    return;
}


void shape_fn(Array2D<double,NODES_PER_ELEMENT>& shape,
		double eta1, double eta2, double eta3)
{
    std::vector<double> lN(NODES_PER_ELEMENT);
    lN[7] = 0.125 * (1.0+eta1) * (1.0-eta2) * (1.0+eta3);
    lN[6] = 0.125 * (1.0+eta1) * (1.0+eta2) * (1.0+eta3);
    lN[5] = 0.125 * (1.0-eta1) * (1.0+eta2) * (1.0+eta3);
    lN[4] = 0.125 * (1.0-eta1) * (1.0-eta2) * (1.0+eta3);
    lN[3] = 0.125 * (1.0+eta1) * (1.0-eta2) * (1.0-eta3);
    lN[2] = 0.125 * (1.0+eta1) * (1.0+eta2) * (1.0-eta3);
    lN[1] = 0.125 * (1.0-eta1) * (1.0+eta2) * (1.0-eta3);
    lN[0] = 0.125 * (1.0-eta1) * (1.0-eta2) * (1.0-eta3);

    shape.push_back(lN);
}
#endif


// version
// $Id: Interpolator.cc,v 1.12 2004/01/08 02:29:37 tan2 Exp $

// End of file
