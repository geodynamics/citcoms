/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *<LicenseText>
 *=====================================================================
 *
 *                              CitcomS
 *                 ---------------------------------
 *
 *                              Authors:
 *           Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 *           Clint Conrad, Michael Gurnis, and Eun-seo Choi
 *          (c) California Institute of Technology 1994-2005
 *
 *        By downloading and/or installing this software you have
 *       agreed to the CitcomS.py-LICENSE bundled with this software.
 *             Free for non-commercial academic research ONLY.
 *      This program is distributed WITHOUT ANY WARRANTY whatsoever.
 *
 *=====================================================================
 *
 *  Copyright June 2005, by the California Institute of Technology.
 *  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
 * 
 *  Any commercial use must be negotiated with the Office of Technology
 *  Transfer at the California Institute of Technology. This software
 *  may be subject to U.S. export control laws and regulations. By
 *  accepting this software, the user agrees to comply with all
 *  applicable U.S. export laws and regulations, including the
 *  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
 *  the Export Administration Regulations, 15 C.F.R. 730-744. User has
 *  the responsibility to obtain export licenses, or other export
 *  authority as may be required before exporting such information to
 *  foreign countries or providing access to foreign nationals.  In no
 *  event shall the California Institute of Technology be liable to any
 *  party for direct, indirect, special, incidental or consequential
 *  damages, including lost profits, arising out of the use of this
 *  software and its documentation, even if the California Institute of
 *  Technology has been advised of the possibility of such damage.
 * 
 *  The California Institute of Technology specifically disclaims any
 *  warranties, including the implied warranties or merchantability and
 *  fitness for a particular purpose. The software and documentation
 *  provided hereunder is on an "as is" basis, and the California
 *  Institute of Technology has no obligations to provide maintenance,
 *  support, updates, enhancements or modifications.
 *
 *=====================================================================
 *</LicenseText>
 * 
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
/*
  This file contains functions that are no longer used in this version of
  CitcomS. To reduce compilantion time and maintanance effort, these functions
  are removed from its original location to here.
*/



/* ==========================================================  */
/* from Size_does_matter.c                                     */
/* =========================================================== */


void get_global_1d_shape_fn_1(E,el,GM,dGammax,nodal,m)
     struct All_variables *E;
     int el,nodal,m;
     struct Shape_function *GM;
     struct Shape_function_dA *dGammax;
{
  int i,k,d,e,h,l,kk;

  double jacobian;
  double determinant();
  double cofactor();
  double **dmatrix();

  const int dims=E->mesh.nsd,dofs=E->mesh.dof;
  const int ends=enodes[dims];

  double dxda[4][4],cof[4][4];


   for(k=1;k<=vpoints[E->mesh.nsd];k++)  {

      for(d=1;d<=dims;d++)
        for(e=1;e<=dims;e++)  {
          dxda[d][e] = 0.0;
          for(i=1;i<=ends;i++)
            dxda[d][e] += E->NMx.vpt[GNVXINDEX(d-1,i,k)]
                * E->x[m][e][E->ien[m][el].node[i]];
          }

      for(d=1;d<=dims;d++)
        for(e=1;e<=dims;e++)   {
          cof[d][e] = 0.0;
          for(h=1;h<=dims;h++)
            cof[d][e] += dxda[d][h]*dxda[e][h];
          }

      if (cof[3][3]!=0.0)
        jacobian = sqrt(abs(determinant(cof,E->mesh.nsd)))/cof[3][3];

      dGammax->vpt[k] = jacobian;

      }

  return;
}


/*   ======================================================================
     For calculating pressure boundary term --- Choi, 11/13/02
     ======================================================================  */
void get_global_side_1d_shape_fn(E,el,GM,GMx,dGamma,NS,far,m)
     struct All_variables *E;
     int el,far,m,NS;
     struct Shape_function1 *GM;
     struct Shape_function1_dx *GMx;
     struct Shape_function_side_dA *dGamma;
{
  int ii,i,j,k,d,a,e,node;

  double jacobian;
  double determinant();
  double cofactor();
  void   form_rtf_bc();

  struct Shape_function1 LGM;
  struct Shape_function1_dx LGMx;

  int dims[2][3];
  int *elist[3];
  const int oned = onedvpoints[E->mesh.nsd];
  const int vpts = vpoints[E->mesh.nsd-1];
  const int ppts = ppoints[E->mesh.nsd-1];
  const int ends = enodes[E->mesh.nsd-1];
  double to,fo,ro,xx[4][5],dxda[4][4],dxdy[4][4];

  /******************************************/
  elist[0] = (int *)malloc(9*sizeof(int));
  elist[1] = (int *)malloc(9*sizeof(int));
  elist[2] = (int *)malloc(9*sizeof(int));
  /*for NS boundary elements */
  elist[0][0]=0; elist[0][1]=1; elist[0][2]=4; elist[0][3]=8; elist[0][4]=5;
  elist[0][5]=2; elist[0][6]=3; elist[0][7]=7; elist[0][8]=6;
  /*for EW boundary elements */
  elist[1][0]=0; elist[1][1]=1; elist[1][2]=2; elist[1][3]=6; elist[1][4]=5;
  elist[1][5]=4; elist[1][6]=3; elist[1][7]=7; elist[1][8]=8;
  /*for TB boundary elements */
  elist[2][0]=0; elist[2][1]=1; elist[2][2]=2; elist[2][3]=3; elist[2][4]=4;
  elist[2][5]=5; elist[2][6]=6; elist[2][7]=7; elist[2][8]=8;
  /******************************************/

  to = E->eco[m][el].centre[1];
  fo = E->eco[m][el].centre[2];
  ro = E->eco[m][el].centre[3];

  dxdy[1][1] = cos(to)*cos(fo);
  dxdy[1][2] = cos(to)*sin(fo);
  dxdy[1][3] = -sin(to);
  dxdy[2][1] = -sin(fo);
  dxdy[2][2] = cos(fo);
  dxdy[2][3] = 0.0;
  dxdy[3][1] = sin(to)*cos(fo);
  dxdy[3][2] = sin(to)*sin(fo);
  dxdy[3][3] = cos(to);

  /*for side elements*/
  for(i=1;i<=ends;i++) {
    a = elist[NS][i+far*ends];
    node=E->ien[m][el].node[a];
    xx[1][i] = E->x[m][1][node]*dxdy[1][1]
      + E->x[m][2][node]*dxdy[1][2]
      + E->x[m][3][node]*dxdy[1][3];
    xx[2][i] = E->x[m][1][node]*dxdy[2][1]
      + E->x[m][2][node]*dxdy[2][2]
      + E->x[m][3][node]*dxdy[2][3];
    xx[3][i] = E->x[m][1][node]*dxdy[3][1]
      + E->x[m][2][node]*dxdy[3][2]
      + E->x[m][3][node]*dxdy[3][3];
  }

  for(k=1;k<=oned;k++)    {
    for(d=1;d<=E->mesh.nsd-1;d++)
      for(e=1;e<=E->mesh.nsd-1;e++)
	dxda[d][e]=0.0;

    if(NS==0) {
      for(i=1;i<=oned;i++) {
	dims[NS][1]=2; dims[NS][2]=3;
	for(d=1;d<=E->mesh.nsd-1;d++)
	  for(e=1;e<=E->mesh.nsd-1;e++) {
	    dxda[d][e] += xx[dims[NS][e]][i]*E->Mx.vpt[GMVXINDEX(d-1,i,k)];
	  }
      }
    }
    else if(NS==1) {
      for(i=1;i<=oned;i++) {
	dims[NS][1]=1; dims[NS][2]=3;
	for(d=1;d<=E->mesh.nsd-1;d++)
	  for(e=1;e<=E->mesh.nsd-1;e++) {
	    dxda[d][e] += xx[dims[NS][e]][i]*E->Mx.vpt[GMVXINDEX(d-1,i,k)];
	  }
      }
    }
    else if(NS==2) {
      for(i=1;i<=oned;i++) {
	dims[NS][1]=1; dims[NS][2]=2;
	for(d=1;d<=E->mesh.nsd-1;d++)
	  for(e=1;e<=E->mesh.nsd-1;e++) {
	    dxda[d][e] += xx[dims[NS][e]][i]*E->Mx.vpt[GMVXINDEX(d-1,i,k)];
	  }
      }
    }

    jacobian = determinant(dxda,E->mesh.nsd-1);
    dGamma->vpt[k] = jacobian;
  }

  for(i=1;i<=ppts;i++)    { /* all of the ppoints*/
    for(d=1;d<=E->mesh.nsd-1;d++)
      for(e=1;e<=E->mesh.nsd-1;e++)
	dxda[d][e]=0.0;

    if(NS==0) {
      for(k=1;k<=ends;k++) {
	dims[NS][1]=2; dims[NS][2]=3;
	for(d=1;d<=E->mesh.nsd-1;d++)
	  for(e=1;e<=E->mesh.nsd-1;e++)
	    dxda[d][e] += xx[dims[NS][e]][k]*E->Mx.ppt[GMPXINDEX(d-1,k,i)];
      }
    }
    else if(NS==1) {
      for(k=1;k<=ends;k++) {
	dims[NS][1]=1; dims[NS][2]=3;
	for(d=1;d<=E->mesh.nsd-1;d++)
	  for(e=1;e<=E->mesh.nsd-1;e++) {
	    a = elist[NS][k+far*ends];
	    node=E->ien[m][el].node[a];
	    dxda[d][e] += xx[dims[NS][e]][k]*E->Mx.ppt[GMPXINDEX(d-1,k,i)];
	  }
      }
    }
    else if(NS==2) {
      for(k=1;k<=ends;k++) {
	dims[NS][1]=1; dims[NS][2]=2;
	for(d=1;d<=E->mesh.nsd-1;d++)
	  for(e=1;e<=E->mesh.nsd-1;e++) {
	    a = elist[NS][k+far*ends];
	    node=E->ien[m][el].node[a];
	    dxda[d][e] += xx[dims[NS][e]][k]*E->Mx.ppt[GMPXINDEX(d-1,k,i)];
	  }
      }
    }

    jacobian = determinant(dxda,E->mesh.nsd-1);
    dGamma->ppt[i] = jacobian;
  }

  for(i=0;i<3;i++)
    free((void *) elist[i]);

  return;
}



/* ==========================================================  */
/* from Element_calculations.c                                 */
/* =========================================================== */

/* ===============================================================
   Function to create the element pressure-forcing vector (due
   to imposed velocity boundary conditions, mixed method).
   =============================================================== */

void get_elt_h(E,el,elt_h,m)
     struct All_variables *E;
     int el,m;
     double elt_h[1];
{
    int i,p,a,b,q,got_g;
    unsigned int type;
    double elt_g[24][1];
    void get_elt_g();

    for(p=0;p<1;p++) elt_h[p] = 0.0;

    got_g = 0;

  type=VBX;
  for(i=1;i<=E->mesh.nsd;i++)
    { for(a=1;a<=enodes[E->mesh.nsd];a++)
	{ if (E->node[m][E->ien[m][el].node[a]] & type)
	    { if(!got_g)
		{  get_elt_g(E,el,elt_g,E->mesh.levmax,m);
		   got_g++;
		 }

	      p=E->mesh.nsd*(a-1) + i - 1;
	      for(b=1;b<=pnodes[E->mesh.nsd];b++)
		{ q = b-1;
		  elt_h[q] -= elt_g[p][q] * E->sphere.cap[m].VB[i][E->ien[m][el].node[a]];
		}
	    }
	}
      type *= (unsigned int) 2;
    }
   return;
}

/* ==========================================================  */
/* from Process_velocity.c                                     */
/* =========================================================== */

void get_ele_visc(E, EV,m)
  struct All_variables *E;
  float *EV;
  int m;
  {

  int el,j,lev;

  const int nel=E->lmesh.nel;
  const int vpts=vpoints[E->mesh.nsd];

  lev = E->mesh.levmax;

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for (el=1;el<=nel;el++)   {
      EV[el] = 0.0;
      for (j=1;j<=vpts;j++)   {
        EV[el] +=  E->EVI[lev][m][(el-1)*vpts+j];
      }

      EV[el] /= vpts;
      }

  return;
  }


/* ==========================================================  */
/*                                                             */
/* =========================================================== */


/* version */
/* $Id: Obsolete.c,v 1.4 2005/06/10 02:23:15 leif Exp $ */

/* End of file  */
