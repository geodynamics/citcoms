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
#include "Boundary.h"
#include "TractionInterpolator.h"

extern "C" {

#include "element_definitions.h"

    void construct_side_c3x3matrix_el(const struct All_variables*, int,
				      struct CC*, struct CCX*,
				      int lev,int m,int pressure,int side);
    void get_global_side_1d_shape_fn(const struct All_variables*, int,
				     struct Shape_function1*,
				     struct Shape_function1_dx*,
				     struct Shape_function_side_dA*,
				     int side, int m);
}


TractionInterpolator::TractionInterpolator(const Boundary& bdry,
					   const All_variables* E,
					   Array2D<int,1>& meshNode) :
    FEMInterpolator(bdry, E, meshNode),
    boundary(bdry),
    meshnode(meshNode),
    gtraction(E->lmesh.nno+1)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;
}


TractionInterpolator::~TractionInterpolator()
{}


void TractionInterpolator::interpolateTraction(Array2D<double,DIM>& target)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

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
    }
}


void TractionInterpolator::interpolateVelocity(Array2D<double,DIM>& V)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    // method copied from VTInterpolator class

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


/////////////////////////////////////////////////////////
// private functions

void TractionInterpolator::computeTraction()
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__) << journal::end;

    //get_global_stress(E);

    const int lev = E->mesh.levmax;
    const int mm = 1;
    for(int i=0; i<size(); ++i) {
	int el = elem_[0][i];
	int n = meshnode[0][i];
	for(int d=0; d<DIM; ++d) {
 	    if(boundary.normal(d,n) == -1)
 		get_elt_traction(el, 0, d, lev, mm);
 	    if(boundary.normal(d,n) == 1)
 		get_elt_traction(el, 1, d, lev, mm);
	}
    }
}


void TractionInterpolator::get_global_stress(const All_variables* E)
{
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
    int side[3][2] = {{SIDE_NORTH, SIDE_SOUTH},
		      {SIDE_WEST, SIDE_EAST},
		      {SIDE_BOTTOM, SIDE_TOP}};

    double traction[4][5],traction_at_gs[4][5];
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
    elist[1][0]=0; elist[1][1]=1; elist[1][2]=5; elist[1][3]=6; elist[1][4]=2;
    elist[1][5]=4; elist[1][6]=8; elist[1][7]=7; elist[1][8]=3;
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

    construct_side_c3x3matrix_el(E,el,&Cc,&Ccx,lev,m,0,side[NS][far]);
    get_global_side_1d_shape_fn(E,el,&GM,&GMx,&dGamma,side[NS][far],m);

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
	    traction[2][j] = E->gstress[m][(a-1)*6+2];
	    traction[3][j] = E->gstress[m][(a-1)*6+6];
	}
    else if(NS==2)
	for(j=1;j<=ends;j++) {
	    a = E->ien[m][el].node[elist[NS][j+ends*far]];
	    traction[1][j] = E->gstress[m][(a-1)*6+5];
	    traction[2][j] = E->gstress[m][(a-1)*6+6];
	    traction[3][j] = E->gstress[m][(a-1)*6+3];
	}
    else {
	std::cout << " NS value is wrong!!" << std::endl;
	exit(0);
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
	a=elist[NS][e+ends*far];
	p=E->ien[m][el].node[a];
	for(i=1;i<=dims;i++)  {
	    x[i]=0.0;
	    for(k=1;k<=vpts;k++) {
		// in 2D 4-pt Gauss quadrature, the weighting factor is 1.
		temp = 1.0 * dGamma.vpt[k];
		x[i]+=E->M.vpt[GMVINDEX(e,k)]*temp*(Cc.vpt[BVINDEX(1,i,a,k)]*traction_at_gs[1][k]
						    +Cc.vpt[BVINDEX(2,i,a,k)]*traction_at_gs[2][k]
						    +Cc.vpt[BVINDEX(3,i,a,k)]*traction_at_gs[3][k]);
	    }
	    gtraction[i-1][p] += x[i];
	}
    }

    for(int i=0;i<dims;i++)
	delete [] elist[i];

    return;
}


// version
// $Id: TractionInterpolator.cc,v 1.12 2004/04/15 18:39:54 tan2 Exp $

// End of file
