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
#include "TractionInterpolator.h"

extern "C" {

#include "element_definitions.h"

    void construct_side_c3x3matrix_el(const struct All_variables*,int,struct CC,struct CCX,int,int,int,int,int);
    void get_global_side_1d_shape_fn(const struct All_variables*,int,struct Shape_function1,struct Shape_function1_dx,struct Shape_function_side_dA,int,int,int);

}


TractionInterpolator::TractionInterpolator(const BoundedMesh& boundedMesh,
					   Array2D<int,1>& meshNode,
					   const All_variables* e) :
    FEMInterpolator(0),
    E(e)
{
    init(boundedMesh, meshNode);
    //selfTest(boundedMesh, meshNode);
    initComputeTraction(boundedMesh);

    // for the time being, domain_cutout is hidden here.
    domain_cutout();

    elem_.print("elem");
    //shape_.print("shape");
}


TractionInterpolator::~TractionInterpolator()
{
    for(int j=0; j<DIM; j++)
	delete [] gtraction[j];
}


void TractionInterpolator::InterpolateTraction(Array2D<double,DIM>& target)
{
    target.assign(size(), 0);

    computeTraction();

    const int mm = 1;
    for(int i=0; i<size(); i++) {
	int n1 = elem_[0][i];
	for(int k=0; k< NODES_PER_ELEMENT; k++) {
	    int node = E->ien[mm][n1].node[k+1];
	    for(int d=0; d<DIM; d++) {
		target[d][i] += shape_[k][i] * gtraction[d][node];
	    }
	}
// 	std::cout << target[0][i] << " " << target[1][i] << " "
// 		  << target[2][i] << std::endl;
    }
}

void TractionInterpolator::domain_cutout()
{

    int m=1;

    // cut out elements only when a part of the embedded domain is included
    int isRelevant=do_xmin+do_xmax+do_ymin+do_ymax+do_zmin+do_zmax;
    std::cout << "Elements cut_out: " << isRelevant << " "
	      << do_xmin << " " << do_xmax << " " << do_ymin << " "
	      << do_ymax << " " << do_zmin << " " << do_zmax << " "
	      << dm_xmin << " " << dm_xmax << " " << dm_ymin << " "
	      << dm_ymax << " " << dm_zmin << " " << dm_zmax << std::endl;

    if(isRelevant) {
	for(int i=dm_ymin;i<=dm_ymax;i++)
	    for(int j=dm_xmin;j<=dm_xmax;j++)
		for(int k=dm_zmin;k<=dm_zmax;k++) {
		    int kk=k+E->lmesh.ezs;
		    int jj=j+E->lmesh.exs;
		    int ii=i+E->lmesh.eys;
		    int el=kk+(jj-1)*E->lmesh.elz+(ii-1)*E->lmesh.elz*E->lmesh.elx;
		    E->element[m][el]=1;

		    std::cout << "Elements cut out: " << el << " "
			      << k << " " << j << " " << i << std::endl;
		}
    }
}

/////////////////////////////////////////////////////////
// private functions


// vertices of five sub-tetrahedra
const int nsub[] = {0, 2, 3, 7,
		    0, 1, 2, 5,
		    4, 7, 5, 0,
		    5, 7, 6, 2,
		    5, 7, 2, 0};

void TractionInterpolator::init(const BoundedMesh& boundedMesh,
				Array2D<int,1>& meshNode)
{
    double xt[DIM], xc[DIM*NODES_PER_ELEMENT], x1[DIM], x2[DIM], x3[DIM], x4[DIM];

    findMaxGridSpacing();

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
		    journal::firewall_t firewall("TractionInterpolator");
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

		// found if all det are greater than zero
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
// 	    if(found) break;
	}
    }
    elem_.shrink();
    shape_.shrink();
}


void TractionInterpolator::findMaxGridSpacing()
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


bool TractionInterpolator::isCandidate(const double* xc,
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


double TractionInterpolator::TetrahedronVolume(double *x1, double *x2,
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


double TractionInterpolator::det3_sub(double *x1, double *x2, double *x3) const
{
    return (x1[0]*(x2[1]*x3[2]-x3[1]*x2[2])
            -x1[1]*(x2[0]*x3[2]-x3[0]*x2[2])
            +x1[2]*(x2[0]*x3[1]-x3[0]*x2[1]));
}


void TractionInterpolator::appendFoundElement(int el, int ntetra,
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


void TractionInterpolator::get_elt_traction(int el,
					    int far,
					    int NS,
					    int lev,
					    int m)
{
    int i,p,a,j,k,e;
    int elx,ely,elz;
    double temp,x[4];
    int *elist[3];

    float traction[4][5],traction_at_gs[4][5];
    struct Shape_function1 GM;
    struct Shape_function_side_dA dGamma;
    struct Shape_function1_dx GMx;
    struct CC Cc;
    struct CCX Ccx;

    const int dims=E->mesh.nsd;
    const int ends=enodes[dims-1];
    const int vpts=onedvpoints[E->mesh.nsd];

    for(int i=0;i<dims;i++)
	elist[i] = new int[9];

    // for NS boundary elements
    elist[0][0]=0; elist[0][1]=1; elist[0][2]=4; elist[0][3]=8; elist[0][4]=5;
    elist[0][5]=2; elist[0][6]=3; elist[0][7]=7; elist[0][8]=6;
    // for EW boundary elements
    elist[1][0]=0; elist[1][1]=1; elist[1][2]=2; elist[1][3]=6; elist[1][4]=5;
    elist[1][5]=4; elist[1][6]=3; elist[1][7]=7; elist[1][8]=8;
    // for TB boundary elements
    elist[2][0]=0; elist[2][1]=1; elist[2][2]=2; elist[2][3]=3; elist[2][4]=4;
    elist[2][5]=5; elist[2][6]=6; elist[2][7]=7; elist[2][8]=8;

    elx=E->lmesh.nox;
    ely=E->lmesh.noy;
    elz=E->lmesh.noz;

    for(i=0;i<=dims;i++) {
	x[i]=0.0;
	for(j=0;j<=ends;j++) {
	    traction[i][j] = 0.0;
	    traction_at_gs[i][j] = 0.0;
	}
    }
    // Test
    std::cout << "get_elt_traction: " << "el=" << el << " "
	      << "lev=" << lev << " " << "m=" << m << " " << "NS=" << NS
	      << " " << "far=" << far << std::endl;
    //

    construct_side_c3x3matrix_el(E,el,Cc,Ccx,lev,m,0,NS,far);
    get_global_side_1d_shape_fn(E,el,GM,GMx,dGamma,NS,far,m);


    // if normal is in theta direction: 0, in fi: 1, and in r: 2
    if(NS==0)
	for(j=1;j<=ends;j++) {
	    a = E->ien[m][el].node[elist[NS][j+ends*far]];
	    traction[1][j] = E->gstress[m][(a-1)*6+1];
	    traction[2][j] = E->gstress[m][(a-1)*6+4];
	    traction[3][j] = E->gstress[m][(a-1)*6+5];
	}
    else if(NS==1)
	for(j=1;j<=ends;j++) {
	    a = E->ien[m][el].node[elist[NS][j+ends*far]];
	    traction[1][j] = E->gstress[m][(a-1)*6+4];
	    traction[2][j] = E->gstress[m][(a-1)*6+3];
	    traction[3][j] = E->gstress[m][(a-1)*6+6];
	}
    else if(NS==2)
	for(j=1;j<=ends;j++) {
	    a = E->ien[m][el].node[elist[NS][j+ends*far]];
	    traction[1][j] = E->gstress[m][(a-1)*6+5];
	    traction[2][j] = E->gstress[m][(a-1)*6+6];
	    traction[3][j] = E->gstress[m][(a-1)*6+2];
	}

    // seems ad hoc: improve later
    if(far==0 && (NS==0 || NS==1)) {
	for(i=1;i<=dims;i++)
	    for(j=1;j<=ends;j++)
		traction[i][j] *= -1.0;
    }

    if(NS==2 && far==1) {
	for(i=1;i<=dims;i++)
	    for(j=1;j<=ends;j++) {
		traction[i][j] *= -1.0;
	    }
    }
    //

    for(k=1;k<=vpts;k++) {
	for(e=1;e<=ends;e++) {
	    traction_at_gs[1][k] += E->M.vpt[GMVINDEX(e,k)]*traction[1][e];
	    traction_at_gs[2][k] += E->M.vpt[GMVINDEX(e,k)]*traction[2][e];
	    traction_at_gs[3][k] += E->M.vpt[GMVINDEX(e,k)]*traction[3][e];
	}
    }

    for(e=1;e<=ends;e++) {
	for(i=1;i<=dims;i++)  {
	    a=elist[NS][e+ends*far];
	    for(k=1;k<=vpts;k++) {
		// in 2D 4-pt Gauss quadrature, the weighting factor is 1.
		temp = 1.0 * dGamma.vpt[k];
		x[i]+=E->M.vpt[GMVINDEX(e,k)]*temp*(Cc.vpt[BVINDEX(1,i,a,k)]*traction_at_gs[1][k]
						    +Cc.vpt[BVINDEX(2,i,a,k)]*traction_at_gs[2][k]
						    +Cc.vpt[BVINDEX(3,i,a,k)]*traction_at_gs[3][k]);
	    }
	    p=E->ien[m][el].node[a];
	    gtraction[i-1][p] += x[i];
	}
    }

    for(int i=0;i<dims;i++)
	delete [] elist[i];

    return;
}


void TractionInterpolator::initComputeTraction(const BoundedMesh& boundedMesh)
{

    for(int j=0; j<DIM; j++)
	gtraction[j] = new float[E->lmesh.nno+1];

    const BoundedBox& bbox = boundedMesh.bbox();

    int elx=E->lmesh.elx;
    int ely=E->lmesh.ely;
    int elz=E->lmesh.elz;
    int nox=E->lmesh.nox;
    int noy=E->lmesh.noy;
    int noz=E->lmesh.noz;

    dm_xmin=xmin=1;
    dm_xmax=xmax=elx;
    dm_ymin=ymin=1;
    dm_ymax=ymax=ely;
    dm_zmin=zmin=1;
    dm_zmax=zmax=elz;

    do_xmin=0;
    do_xmax=0;
    do_ymin=0;
    do_ymax=0;
    do_zmin=0;
    do_zmax=0;

    int m=1;
    for(int i=1;i<=E->lmesh.noy;i++)
	for(int j=1;j<=E->lmesh.nox;j++)
	    for(int k=1;k<=E->lmesh.noz;k++) {
		int node=k+(j-1)*E->lmesh.noz+(i-1)*E->lmesh.noz*E->lmesh.nox;
		if(E->sx[m][1][node]>=bbox[0][0]-1.0e-6 &&
		   E->sx[m][1][node]<=bbox[0][0]+1.0e-6 &&
		   E->sx[m][2][node]>=bbox[0][1]-1.0e-6 &&
		   E->sx[m][2][node]<=bbox[0][1]+1.0e-6 &&
		   E->sx[m][3][node]>=bbox[0][2]-1.0e-6 &&
		   E->sx[m][3][node]<=bbox[0][2]+1.0e-6) {
		    if(j>1 && i>1 && k>1) {
			if(!do_xmin) {
			    xmin=j-1;
			    dm_xmin=j;
			    do_xmin=1;
			}
			if(!do_ymin) {
			    ymin=i-1;
			    dm_ymin=i;
			    do_ymin=1;
			}
			if(!do_zmin) {
			    zmin=k-1;
			    dm_zmin=k;
			    do_zmin=1;
			}
		    }
		}
		if(E->sx[m][1][node]>=bbox[1][0]-1.0e-6 &&
		   E->sx[m][1][node]<=bbox[1][0]+1.0e-6 &&
		   E->sx[m][2][node]>=bbox[0][1]-1.0e-6 &&
		   E->sx[m][2][node]<=bbox[0][1]+1.0e-6 &&
		   E->sx[m][3][node]>=bbox[0][2]-1.0e-6 &&
		   E->sx[m][3][node]<=bbox[0][2]+1.0e-6) {
		    if(j<nox && i>1 && k>1) {
			if(!do_xmax) {
			    xmax=j;
			    dm_xmax=j-1;
			    do_xmax=1;
			}
			if(!do_ymin) {
			    ymin=i-1;
			    dm_ymin=i;
			    do_ymin=1;
			}
			if(!do_zmin) {
			    zmin=k-1;
			    dm_zmin=k;
			    do_zmin=1;
			}
		    }
		}
		if(E->sx[m][1][node]>=bbox[0][0]-1.0e-6 &&
		   E->sx[m][1][node]<=bbox[0][0]+1.0e-6 &&
		   E->sx[m][2][node]>=bbox[1][1]-1.0e-6 &&
		   E->sx[m][2][node]<=bbox[1][1]+1.0e-6 &&
		   E->sx[m][3][node]>=bbox[0][2]-1.0e-6 &&
		   E->sx[m][3][node]<=bbox[0][2]+1.0e-6) {
		    if(j>1 && i<noy && k>1) {
			if(!do_xmin) {
			    xmin=j-1;
			    dm_xmin=j;
			    do_xmin=1;
			}
			if(!do_ymax) {
			    ymax=i;
			    dm_ymax=i-1;
			    do_ymax=1;
			}
			if(!do_zmin) {
			    zmin=k-1;
			    dm_zmin=k;
			    do_zmin=1;
			}
		    }
		}
		if(E->sx[m][1][node]>=bbox[1][0]-1.0e-6 &&
		   E->sx[m][1][node]<=bbox[1][0]+1.0e-6 &&
		   E->sx[m][2][node]>=bbox[1][1]-1.0e-6 &&
		   E->sx[m][2][node]<=bbox[1][1]+1.0e-6 &&
		   E->sx[m][3][node]>=bbox[0][2]-1.0e-6 &&
		   E->sx[m][3][node]<=bbox[0][2]+1.0e-6) {
		    if(j<nox && i<noy && k>1) {
			if(!do_xmax) {
			    xmax=j;
			    dm_xmax=j-1;
			    do_xmax=1;
			}
			if(!do_ymax) {
			    ymax=i;
			    dm_ymax=i-1;
			    do_ymax=1;
			}
			if(!do_zmin) {
			    zmin=k-1;
			    dm_zmin=k;
			    do_zmin=1;
			}
		    }
		}
		if(E->sx[m][1][node]>=bbox[0][0]-1.0e-6 &&
		   E->sx[m][1][node]<=bbox[0][0]+1.0e-6 &&
		   E->sx[m][2][node]>=bbox[0][1]-1.0e-6 &&
		   E->sx[m][2][node]<=bbox[0][1]+1.0e-6 &&
		   E->sx[m][3][node]>=bbox[1][2]-1.0e-6 &&
		   E->sx[m][3][node]<=bbox[1][2]+1.0e-6) {
		    if(j>1 && i>1 && k<noz) {
			if(!do_xmin) {
			    xmin=j-1;
			    dm_xmin=j;
			    do_xmin=1;
			}
			if(!do_ymin) {
			    ymin=i-1;
			    dm_ymin=i;
			    do_ymin=1;
			}
			if(!do_zmax) {
			    zmax=k;
			    dm_zmax=k-1;
			    do_zmax=1;
			}
		    }
		}
		if(E->sx[m][1][node]>=bbox[1][0]-1.0e-6 &&
		   E->sx[m][1][node]<=bbox[1][0]+1.0e-6 &&
		   E->sx[m][2][node]>=bbox[0][1]-1.0e-6 &&
		   E->sx[m][2][node]<=bbox[0][1]+1.0e-6 &&
		   E->sx[m][3][node]>=bbox[1][2]-1.0e-6 &&
		   E->sx[m][3][node]<=bbox[1][2]+1.0e-6) {
		    if(j<nox && i>1 && k<noz) {
			if(!do_xmax) {
			    xmax=j;
			    dm_xmax=j-1;
			    do_xmax=1;
			}
			if(!do_ymin) {
			    ymin=i-1;
			    dm_ymin=i;
			    do_ymin=1;
			}
			if(!do_zmax) {
			    zmax=k;
			    dm_zmax=k-1;
			    do_zmax=1;
			}
		    }
		}
		if(E->sx[m][1][node]>=bbox[0][0]-1.0e-6 &&
		   E->sx[m][1][node]<=bbox[0][0]+1.0e-6 &&
		   E->sx[m][2][node]>=bbox[1][1]-1.0e-6 &&
		   E->sx[m][2][node]<=bbox[1][1]+1.0e-6 &&
		   E->sx[m][3][node]>=bbox[1][2]-1.0e-6 &&
		   E->sx[m][3][node]<=bbox[1][2]+1.0e-6) {
		    if(j>1 && i<noy && k<noz) {
			if(!do_xmin) {
			    xmin=j-1;
			    dm_xmin=j;
			    do_xmin=1;
			}
			if(!do_ymax) {
			    ymax=i;
			    dm_ymax=i-1;
			    do_ymax=1;
			}
			if(!do_zmax) {
			    zmax=k;
			    dm_zmax=k-1;
			    do_zmax=1;
			}
		    }
		}
		if(E->sx[m][1][node]>=bbox[1][0]-1.0e-6 &&
		   E->sx[m][1][node]<=bbox[1][0]+1.0e-6 &&
		   E->sx[m][2][node]>=bbox[1][1]-1.0e-6 &&
		   E->sx[m][2][node]<=bbox[1][1]+1.0e-6 &&
		   E->sx[m][3][node]>=bbox[1][2]-1.0e-6 &&
		   E->sx[m][3][node]<=bbox[1][2]+1.0e-6) {
		    if(j<nox && i<noy && k<noz) {
			if(!do_xmax) {
			    xmax=j;
			    dm_xmax=j-1;
			    do_xmax=1;
			}
			if(!do_ymax) {
			    ymax=i;
			    dm_ymax=i-1;
			    do_ymax=1;
			}
			if(!do_zmax) {
			    zmax=k;
			    dm_zmax=k-1;
			    do_zmax=1;
			}
		    }
		}

	    }

    std::cout << "me=" << E->parallel.me << " " << xmin << " " << xmax << " "
	      << ymin << " " << ymax << " "
	      << zmin << " " << zmax << " "
	      << do_xmin << " " << do_xmax << " "
	      << do_ymin << " " << do_ymax << " "
	      << do_zmin << " " << do_zmax << " " << std::endl;
    return;
}


void TractionInterpolator::computeTraction()
{

    int elx=E->lmesh.elx;
    int elz=E->lmesh.elz;

    int lev=E->mesh.levmax;
    int mm=1;
    // north/south

    for(int i=ymin;i<=ymax;i++)
	for(int j=zmin;j<=elz;j++) {
	    if(do_xmin) {
		int el=j+(xmin-1)*elz+(i-1)*elz*elx;
		std::cout << "N/S:" << el << std::endl;
		get_elt_traction(el,1,0,lev,mm);
	    }
	    if(do_xmax) {
		int el=j+(xmax-1)*elz+(i-1)*elz*elx;
		std::cout << "N/S:" << el << std::endl;
		get_elt_traction(el,0,0,lev,mm);
	    }
	}
    // west/east
    for(int i=xmin;i<=xmax;i++)
	for(int j=zmin;j<=elz;j++) {
	    if(do_ymin) {
		int el=j+(i-1)*elz+(ymin-1)*elz*elx;
		std::cout << "W/E:" << el << std::endl;
		get_elt_traction(el,1,1,lev,mm);
	    }
	    if(do_ymax) {
		int el=j+(i-1)*elz+(ymax-1)*elz*elx;
		std::cout << "W/E:" << el << std::endl;
		get_elt_traction(el,0,1,lev,mm);
	    }
	}
    // bottom/top
    for(int i=ymin;i<=ymax;i++)
	for(int j=xmin;j<=xmax;j++) {
	    if(do_zmin) {
		int el=zmin+(j-1)*elz+(i-1)*elz*elx;
		std::cout << "B:" << el << std::endl;
		get_elt_traction(el,1,2,lev,mm);
	    }
	}

    return;
}


void TractionInterpolator::selfTest(const BoundedMesh& boundedMesh,
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

	    journal::firewall_t firewall("TractionInterpolator");
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
// $Id: TractionInterpolator.cc,v 1.5 2004/01/08 02:29:37 tan2 Exp $

// End of file
