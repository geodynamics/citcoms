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
					   const All_variables* E,
					   Array2D<int,1>& meshNode) :
    FEMInterpolator(boundedMesh, E, meshNode)
{
    journal::debug_t debug("Exchanger");
    debug << journal::loc(__HERE__)
	  << "in TractionInterpolator::c'tor" << journal::end;

    initComputeTraction(boundedMesh);

    // for the time being, domain_cutout is hidden here.
    domain_cutout();

}


TractionInterpolator::~TractionInterpolator()
{
    for(int j=0; j<DIM; j++)
	delete [] gtraction[j];
}


void TractionInterpolator::interpolateTraction(Array2D<double,DIM>& target)
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

// version
// $Id: TractionInterpolator.cc,v 1.7 2004/01/14 02:40:40 ces74 Exp $

// End of file
