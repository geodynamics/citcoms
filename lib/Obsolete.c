/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 *<LicenseText>
 *
 * CitcomS by Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 * Clint Conrad, Michael Gurnis, and Eun-seo Choi.
 * Copyright (C) 1994-2005, California Institute of Technology.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
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



/*	==================================================================================
	Function to give the global shape function from the local: Assumes ORTHOGONAL MESH
	==================================================================================      */

void get_global_shape_fn(E,el,GN,GNx,dOmega,pressure,sphere,rtf,lev,m)
     struct All_variables *E;
     int el,m;
     struct Shape_function *GN;
     struct Shape_function_dx *GNx;
     struct Shape_function_dA *dOmega;
     int pressure,lev,sphere;
     double rtf[4][9];
{
  int i,j,k,d,e;
  double jacobian;
  double determinant();
  double cofactor(),myatan();
  void   form_rtf_bc();

  struct Shape_function_dx LGNx;

  double dxda[4][4],cof[4][4],x[4],bc[4][4];


  const int dims=E->mesh.nsd;
  const int ends=enodes[dims];
  const int vpts=vpoints[dims];
  const int ppts=ppoints[dims];


  if(pressure < 2) {
    for(k=1;k<=vpts;k++) {       /* all of the vpoints */
      for(d=1;d<=dims;d++)  {
        x[d]=0.0;
        for(e=1;e<=dims;e++)
          dxda[d][e]=0.0;
        }

      for(d=1;d<=dims;d++)
        for(i=1;i<=ends;i++)
          x[d] += E->X[lev][m][d][E->IEN[lev][m][el].node[i]]*
                E->N.vpt[GNVINDEX(i,k)];

      for(d=1;d<=dims;d++)
	for(e=1;e<=dims;e++)
	  for(i=1;i<=ends;i++)
            dxda[d][e] += E->X[lev][m][e][E->IEN[lev][m][el].node[i]]
               * E->Nx.vpt[GNVXINDEX(d-1,i,k)];

      jacobian = determinant(dxda,E->mesh.nsd);
      dOmega->vpt[k] = jacobian;

      for(d=1;d<=dims;d++)
        for(e=1;e<=dims;e++)
          cof[d][e]=cofactor(dxda,d,e,dims);

      if (sphere)   {

        form_rtf_bc(k,x,rtf,bc);
        for(j=1;j<=ends;j++)
          for(d=1;d<=dims;d++)         {
            LGNx.vpt[GNVXINDEX(d-1,j,k)] = 0.0;
            for(e=1;e<=dims;e++)
              LGNx.vpt[GNVXINDEX(d-1,j,k)] +=
                 E->Nx.vpt[GNVXINDEX(e-1,j,k)] *cof[e][d];

            LGNx.vpt[GNVXINDEX(d-1,j,k)] /= jacobian;
            }

        for(j=1;j<=ends;j++)
          for(d=1;d<=dims;d++)         {
            GNx->vpt[GNVXINDEX(d-1,j,k)] =
                bc[d][1]*LGNx.vpt[GNVXINDEX(0,j,k)]
              + bc[d][2]*LGNx.vpt[GNVXINDEX(1,j,k)]
              + bc[d][3]*LGNx.vpt[GNVXINDEX(2,j,k)];
            }
        }
      else  {
        for(j=1;j<=ends;j++)
          for(d=1;d<=dims;d++)         {
            GNx->vpt[GNVXINDEX(d-1,j,k)] = 0.0;
            for(e=1;e<=dims;e++)
              GNx->vpt[GNVXINDEX(d-1,j,k)] +=
                 E->Nx.vpt[GNVXINDEX(e-1,j,k)] *cof[e][d];

            GNx->vpt[GNVXINDEX(d-1,j,k)] /= jacobian;
            }
        }
      }     /* end for k */
    }    /* end for pressure */

  if(pressure > 0 && pressure < 3) {
    for(k=1;k<=ppts;k++)         {   /* all of the ppoints */
      for(d=1;d<=dims;d++) {
        x[d]=0.0;
        for(e=1;e<=dims;e++)
          dxda[d][e]=0.0;
        }

      for(d=1;d<=dims;d++)
        for(i=1;i<=ends;i++)
          x[d] += E->X[lev][m][d][E->IEN[lev][m][el].node[i]]
                 *E->N.ppt[GNPINDEX(i,k)];

      for(d=1;d<=dims;d++)
	for(e=1;e<=dims;e++)
	  for(i=1;i<=ends;i++)
            dxda[d][e] += E->X[lev][m][e][E->IEN[lev][m][el].node[i]]
                     * E->Nx.ppt[GNPXINDEX(d-1,i,k)];

      jacobian = determinant(dxda,E->mesh.nsd);
      dOmega->ppt[k] = jacobian;

      for(d=1;d<=dims;d++)
        for(e=1;e<=dims;e++)
          cof[d][e]=cofactor(dxda,d,e,E->mesh.nsd);

      if (sphere)   {
        form_rtf_bc(k,x,rtf,bc);
        for(j=1;j<=ends;j++)
          for(d=1;d<=dims;d++)  {
            LGNx.ppt[GNPXINDEX(d-1,j,k)]=0.0;
            for(e=1;e<=dims;e++)
              LGNx.ppt[GNPXINDEX(d-1,j,k)] +=
                E->Nx.ppt[GNPXINDEX(e-1,j,k)]*cof[e][d];
	    LGNx.ppt[GNPXINDEX(d-1,j,k)] /= jacobian;
            }
        for(j=1;j<=ends;j++)
          for(d=1;d<=dims;d++)         {
            GNx->ppt[GNPXINDEX(d-1,j,k)]
             = bc[d][1]*LGNx.ppt[GNPXINDEX(0,j,k)]
             + bc[d][2]*LGNx.ppt[GNPXINDEX(1,j,k)]
             + bc[d][3]*LGNx.ppt[GNPXINDEX(2,j,k)];
          }
        }

      else  {
        for(j=1;j<=ends;j++)
          for(d=1;d<=dims;d++)  {
            GNx->ppt[GNPXINDEX(d-1,j,k)]=0.0;
            for(e=1;e<=dims;e++)
              GNx->ppt[GNPXINDEX(d-1,j,k)] +=
                E->Nx.ppt[GNPXINDEX(e-1,j,k)]*cof[e][d];
	    GNx->ppt[GNPXINDEX(d-1,j,k)] /= jacobian;
            }
        }

      }              /* end for k int */
    }      /* end for pressure */


  return;
}


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
        jacobian = sqrt(fabs(determinant(cof,E->mesh.nsd)))/cof[3][3];

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
    higher_precision elt_g[24][1];
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
/*  From Sphere_harmonics.c                                    */
/* =========================================================== */



/* This function construct sph harm tables on a regular grid */
/* for inverse interpolation                                 */

static void  compute_sphereh_int_table(E)
     struct All_variables *E;
{
    int i,j;
    double t,f;
    double dth,dfi,sqrt_multis();

    E->sphere.con = (double *)malloc(E->sphere.hindice*sizeof(double));
    for (ll=0;ll<=E->output.llmax;ll++)
	for (mm=0;mm<=ll;mm++)   {
	    E->sphere.con[E->sphere.hindex[ll][mm]] =
		sqrt( (2.0-((mm==0)?1.0:0.0))*(2*ll+1)/(4.0*M_PI) )
		*sqrt_multis(ll+mm,ll-mm);  /* which is sqrt((ll-mm)!/(ll+mm)!) */
	}

    E->sphere.tablenplm   = (double **) malloc((E->sphere.nox+1)
					       *sizeof(double*));
    for (i=1;i<=E->sphere.nox;i++)
	E->sphere.tablenplm[i]= (double *)malloc(E->sphere.hindice
						 *sizeof(double));

    E->sphere.tablencosf  = (double **) malloc((E->sphere.noy+1)
					       *sizeof(double*));
    E->sphere.tablensinf  = (double **) malloc((E->sphere.noy+1)
					       *sizeof(double*));
    for (i=1;i<=E->sphere.noy;i++)   {
	E->sphere.tablencosf[i]= (double *)malloc((E->output.llmax+3)
						  *sizeof(double));
	E->sphere.tablensinf[i]= (double *)malloc((E->output.llmax+3)
						  *sizeof(double));
    }

    E->sphere.sx[1] = (double *) malloc((E->sphere.nsf+1)*sizeof(double));
    E->sphere.sx[2] = (double *) malloc((E->sphere.nsf+1)*sizeof(double));

    dth = M_PI/E->sphere.elx;
    dfi = 2.0*M_PI/E->sphere.ely;

    for (j=1;j<=E->sphere.noy;j++)
	for (i=1;i<=E->sphere.nox;i++) {
	    node = i+(j-1)*E->sphere.nox;
	    E->sphere.sx[1][node] = dth*(i-1);
	    E->sphere.sx[2][node] = dfi*(j-1);
	}

    for (j=1;j<=E->sphere.nox;j++)  {
	t=E->sphere.sx[1][j];
	for (ll=0;ll<=E->output.llmax;ll++)
	    for (mm=0;mm<=ll;mm++)  {
		p = E->sphere.hindex[ll][mm];
		E->sphere.tablenplm[j][p] = modified_plgndr_a(ll,mm,t) ;
	    }
    }
    for (j=1;j<=E->sphere.noy;j++)  {
	node = 1+(j-1)*E->sphere.nox;
	f=E->sphere.sx[2][node];
	for (mm=0;mm<=E->output.llmax;mm++)   {
	    E->sphere.tablencosf[j][mm] = cos( (double)(mm)*f );
	    E->sphere.tablensinf[j][mm] = sin( (double)(mm)*f );
	}
    }
}


/* ================================================
   for a given node, this routine gives which cap and element
   the node is in.
   ================================================*/
void construct_interp_net(E)
     struct All_variables *E;
{

    void parallel_process_termination();
    void parallel_process_sync();
    int ii,jj,es,i,j,m,el,node;
    int locate_cap(),locate_element();
    double x[4],t,f;

    const int ends=4;

    for (i=1;i<=E->sphere.nox;i++)
	for (j=1;j<=E->sphere.noy;j++)   {
	    node = i+(j-1)*E->sphere.nox;
	    E->sphere.int_cap[node]=0;
	    E->sphere.int_ele[node]=0;
	}


    for (i=1;i<=E->sphere.nox;i++)
	for (j=1;j<=E->sphere.noy;j++)   {
	    node = i+(j-1)*E->sphere.nox;

	    /* first find which cap this node (i,j) is in  */
	    t = E->sphere.sx[1][node];
	    f = E->sphere.sx[2][node];

	    x[1] = sin(t)*cos(f);  /* radius does not matter */
	    x[2] = sin(t)*sin(f);
	    x[3] = cos(t);


	    fprintf(E->fp,"mmm0=%d\n",node);

	    m = locate_cap(E,x);

	    fprintf(E->fp,"mmm=%d\n",m);

	    if (m>0)  {
		el = locate_element(E,m,x,node);        /* bottom element */

		if (el<=0)    {
		    fprintf(stderr,"!!! Processor %d cannot find the right element in cap %d\n",E->parallel.me,m);
		    parallel_process_termination();
		}

		E->sphere.int_cap[node]=m;
		E->sphere.int_ele[node]=el;

	    }
	}        /* end for i and j */

    parallel_process_sync(E);

    return;
}

/* ================================================
   locate the cap for node (i,j)
   ================================================*/

int locate_cap(E,x)
     struct All_variables *E;
     double x[4];
{

    int ia[5],i,m,mm;
    double xx[4],angle[5],angle1[5];
    double get_angle();
    double area1,rr;
    const double e_7=1.e-7;
    static int been_here=0;

    mm = 0;

    for (m=1;m<=E->sphere.caps_per_proc;m++)   {
	ia[1] = 1;
	ia[2] = E->lmesh.noz*E->lmesh.nox-E->lmesh.noz+1;
	ia[3] = E->lmesh.nno-E->lmesh.noz+1;
	ia[4] = ia[3]-E->lmesh.noz*(E->lmesh.nox-1);

	for (i=1;i<=4;i++)  {
	    xx[1] = E->x[m][1][ia[i]]/E->sx[m][3][ia[1]];
	    xx[2] = E->x[m][2][ia[i]]/E->sx[m][3][ia[1]];
	    xx[3] = E->x[m][3][ia[i]]/E->sx[m][3][ia[1]];
	    angle[i] = get_angle(x,xx);    /* get angle between (i,j) and other four*/
	    angle1[i]=E->sphere.angle[m][i];
	}

	area1 = area_of_sphere_triag(angle[1],angle[2],angle1[1])
	    + area_of_sphere_triag(angle[2],angle[3],angle1[2])
	    + area_of_sphere_triag(angle[3],angle[4],angle1[3])
	    + area_of_sphere_triag(angle[4],angle[1],angle1[4]);

	if ( fabs ((area1-E->sphere.area[m])/E->sphere.area[m]) <e_7 ) {
	    mm = m;
	    return (mm);
        }
    }

    return (mm);
}

/* ================================================
   locate the element containing the node (i,j) with coord x.
   The radius is assumed to be 1 in computing the areas.
   NOTE:  The returned element el is for the bottom layer.
   ================================================*/

int locate_element(E,m,x,ne)
     struct All_variables *E;
     double x[4];
int m,ne;
{

    int el_temp,el,es,el_located,level,lev,lev_plus,el_plus,es_plus,i,j,node;
    double area,area1,areamin;
    double area_of_5points();
    const double e_7=1.e-7;
    const double e_6=1.e6;

    el_located = 0;


    level=E->mesh.levmin;
    for (es=1;es<=E->lmesh.SNEL[level];es++)              {

	el = (es-1)*E->lmesh.ELZ[level]+1;
	area1 = area_of_5points (E,level,m,el,x,ne);
	area = E->sphere.area1[level][m][es];

	if(fabs ((area1-area)/area) <e_7 ) {
	    for (lev=E->mesh.levmin;lev<E->mesh.levmax;lev++)  {
		lev_plus = lev + 1;
		j=1;
		areamin = e_6;
		do {
		    el_plus = E->EL[lev][m][el].sub[j];

		    es_plus = (el_plus-1)/E->lmesh.ELZ[lev_plus]+1;

		    area1 = area_of_5points(E,lev_plus,m,el_plus,x,ne);
		    area = E->sphere.area1[lev_plus][m][es_plus];

		    if(fabs(area1-area)<areamin) {
			areamin=fabs(area1-area);
			el_temp = el_plus;
		    }
		    j++;
		}  while (j<5 && fabs((area1-area)/area) > e_7);
		el = el_plus;
		/* if exit with ..>e_7, pick the best one*/
		if (fabs((area1-area)/area) > e_7) el = el_temp;
	    }      /* end for loop lev         */
	    el_located = el;
	}    /* end for if */

	if(el_located)  break;
    }    /* end for es at the coarsest level  */

    return (el_located);
}

/* ===============================================================
   interpolate nodal T's within cap m and element el onto node with
   coordinate x[3] which is derived from a regular mesh and within
   the element el. NOTE the radius of x[3] is the inner radius.
   =============================================================== */

float sphere_interpolate_point(E,T,m,el,x,ne)
     struct All_variables *E;
     float **T;
     double x[4];
int m,el,ne;
{
    double to,fo,y[4],yy[4][5],dxdy[4][4];
    double a1,b1,c1,d1,a2,b2,c2,d2,a,b,c,xx1,yy1,y1,y2;
    float ta,t[5];
    int es,snode,i,j,node;

    const int oned=4;
    const double e_7=1.e-7;
    const double four=4.0;
    const double two=2.0;
    const double one=1.0;
    const double pt25=0.25;

    /* first rotate the coord such that the center of element is
       the pole   */

    es = (el-1)/E->lmesh.elz+1;

    to = E->eco[m][el].centre[1];
    fo = E->eco[m][el].centre[2];

    dxdy[1][1] = cos(to)*cos(fo);
    dxdy[1][2] = cos(to)*sin(fo);
    dxdy[1][3] = -sin(to);
    dxdy[2][1] = -sin(fo);
    dxdy[2][2] = cos(fo);
    dxdy[2][3] = 0.0;
    dxdy[3][1] = sin(to)*cos(fo);
    dxdy[3][2] = sin(to)*sin(fo);
    dxdy[3][3] = cos(to);

    for(i=1;i<=oned;i++) {         /* nodes */
	node = E->ien[m][el].node[i];
	snode = E->sien[m][es].node[i];
	t[i] = T[m][snode];
	for (j=1;j<=E->mesh.nsd;j++)
	    yy[j][i] = E->x[m][1][node]*dxdy[j][1]
                + E->x[m][2][node]*dxdy[j][2]
                + E->x[m][3][node]*dxdy[j][3];
    }

    for (j=1;j<=E->mesh.nsd;j++)
	y[j] = x[1]*dxdy[j][1] + x[2]*dxdy[j][2] + x[3]*dxdy[j][3];

    /* then for node y, determine its coordinates xx1,yy1
       in the parental element in the isoparametric element system*/

    a1 = yy[1][1] + yy[1][2] + yy[1][3] + yy[1][4];
    b1 = yy[1][3] + yy[1][2] - yy[1][1] - yy[1][4];
    c1 = yy[1][3] + yy[1][1] - yy[1][2] - yy[1][4];
    d1 = yy[1][3] + yy[1][4] - yy[1][1] - yy[1][2];
    a2 = yy[2][1] + yy[2][2] + yy[2][3] + yy[2][4];
    b2 = yy[2][3] + yy[2][2] - yy[2][1] - yy[2][4];
    c2 = yy[2][3] + yy[2][1] - yy[2][2] - yy[2][4];
    d2 = yy[2][3] + yy[2][4] - yy[2][1] - yy[2][2];

    a = d2*c1;
    b = a2*c1+b1*d2-d1*c2-d1*b2-four*c1*y[2];
    c=four*c2*y[1]-c2*a1-a1*b2+four*b2*y[1]-four*b1*y[2]+a2*b1;

    if (fabs(a)<e_7)  {
	yy1 = -c/b;
	xx1 = (four*y[1]-a1-d1*yy1)/(b1+c1*yy1);
    }
    else  {
	y1= (-b+sqrt(b*b-four*a*c))/(two*a);
	y2= (-b-sqrt(b*b-four*a*c))/(two*a);
	if (fabs(y1)>fabs(y2))
	    yy1 = y2;
	else
	    yy1 = y1;
	xx1 = (four*y[1]-a1-d1*yy1)/(b1+c1*yy1);
    }

    /* now we can calculate T at x[4] using shape function */

    ta = ((one-xx1)*(one-yy1)*t[1]+(one+xx1)*(one-yy1)*t[2]+
          (one+xx1)*(one+yy1)*t[3]+(one-xx1)*(one+yy1)*t[4])*pt25;

    /*if(fabs(xx1)>1.5 || fabs(yy1)>1.5)fprintf(E->fp_out,"ME= %d %d %d %g %g %g %g %g %g %g\n",ne,m,es,t[1],t[2],t[3],t[4],ta,xx1,yy1);
     */
    return (ta);
}

/* ===================================================================
   do the interpolation on sphere for data T, which is needed for both
   spherical harmonic expansion and graphics
   =================================================================== */

void sphere_interpolate(E,T,TG)
     struct All_variables *E;
     float **T,*TG;
{

    float sphere_interpolate_point();
    void gather_TG_to_me0();
    void parallel_process_termination();

    int ii,jj,es,i,j,m,el,node;
    double x[4],t,f;

    const int ends=4;

    TG[0] = 0.0;
    for (i=1;i<=E->sphere.nox;i++)
	for (j=1;j<=E->sphere.noy;j++)   {
	    node = i+(j-1)*E->sphere.nox;
	    TG[node] = 0.0;
	    /* first find which cap this node (i,j) is in  */

	    m = E->sphere.int_cap[node];
	    el = E->sphere.int_ele[node];

	    if (m>0 && el>0)   {
		t = E->sphere.sx[1][node];
		f = E->sphere.sx[2][node];

		x[1] = E->sx[1][3][1]*sin(t)*cos(f);
		x[2] = E->sx[1][3][1]*sin(t)*sin(f);
		x[3] = E->sx[1][3][1]*cos(t);

		TG[node] = sphere_interpolate_point(E,T,m,el,x,node);

	    }

	}        /* end for i and j */

    gather_TG_to_me0(E,TG);

    return;
}



/* ==========================================================  */
/*  From Phase_change.c                                        */
/* =========================================================== */


void phase_change_410(E,B,B_b)
  struct All_variables *E;
  float **B,**B_b;
{
  int i,j,k,n,ns,m;
  float e_pressure,pt5,one;

  pt5 = 0.5; one=1.0;

  for(m=1;m<=E->sphere.caps_per_proc;m++)     {
    for(i=1;i<=E->lmesh.nno;i++)  {
      e_pressure = (E->sphere.ro-E->sx[m][3][i])-E->viscosity.z410-
            E->control.clapeyron410*(E->T[m][i]-E->control.transT410);

      B[m][i] = pt5*(one+tanh(E->control.width410*e_pressure));
      }

    ns = 0;
    for (k=1;k<=E->lmesh.noy;k++)
      for (j=1;j<=E->lmesh.nox;j++)  {
        ns = ns + 1;
        B_b[m][ns]=0.0;
        for (i=1;i<E->lmesh.noz;i++)   {
          n = (k-1)*E->lmesh.noz*E->lmesh.nox + (j-1)*E->lmesh.noz + i;
          if (B[m][n]>=pt5&&B[m][n+1]<=pt5)
            B_b[m][ns]=(E->sx[m][3][n+1]-E->sx[m][3][n])*(pt5-B[m][n])/(B[m][n+1]-B[m][n])+E->sx[m][3][n];
          }
        }
    }


  return;
  }


void phase_change_670(E,B,B_b)
  struct All_variables *E;
  float **B,**B_b;
{
  int i,j,k,n,ns,m;
  float e_pressure,pt5,one;

  pt5 = 0.5; one=1.0;

  for(m=1;m<=E->sphere.caps_per_proc;m++)     {
    for(i=1;i<=E->lmesh.nno;i++)  {
      e_pressure = (E->sphere.ro-E->sx[m][3][i])-E->viscosity.zlm-
            E->control.clapeyron670*(E->T[m][i]-E->control.transT670);

      B[m][i] = pt5*(one+tanh(E->control.width670*e_pressure));
      }

    ns = 0;
    for (k=1;k<=E->lmesh.noy;k++)
      for (j=1;j<=E->lmesh.nox;j++)  {
        ns = ns + 1;
        B_b[m][ns]=0.0;
        for (i=1;i<E->lmesh.noz;i++)   {
          n = (k-1)*E->lmesh.noz*E->lmesh.nox + (j-1)*E->lmesh.noz + i;
          if (B[m][n]>=pt5&&B[m][n+1]<=pt5)
            B_b[m][ns]=(E->sx[m][3][n+1]-E->sx[m][3][n])*(pt5-B[m][n])/(B[m][n+1]-B[m][n])+E->sx[m][3][n];
          }
        }
    }


  return;
  }


void phase_change_cmb(E,B,B_b)
  struct All_variables *E;
  float **B,**B_b;
{
  int i,j,k,n,ns,m;
  float e_pressure,pt5,one;

  pt5 = 0.5; one=1.0;

  for(m=1;m<=E->sphere.caps_per_proc;m++)     {
    for(i=1;i<=E->lmesh.nno;i++)  {
      e_pressure = (E->sphere.ro-E->sx[m][3][i])-E->viscosity.zcmb-
            E->control.clapeyroncmb*(E->T[m][i]-E->control.transTcmb);

      B[m][i] = pt5*(one+tanh(E->control.widthcmb*e_pressure));
      }

    ns = 0;
    for (k=1;k<=E->lmesh.noy;k++)
      for (j=1;j<=E->lmesh.nox;j++)  {
        ns = ns + 1;
        B_b[m][ns]=0.0;
        for (i=1;i<E->lmesh.noz;i++)   {
          n = (k-1)*E->lmesh.noz*E->lmesh.nox + (j-1)*E->lmesh.noz + i;
          if (B[m][n]>=pt5&&B[m][n+1]<=pt5)
            B_b[m][ns]=(E->sx[m][3][n+1]-E->sx[m][3][n])*(pt5-B[m][n])/(B[m][n+1]-B[m][n])+E->sx[m][3][n];
          }
        }
    }

  return;
}


/* ==========================================================  */
/*  From Nodal_mesh.c                                          */
/* =========================================================== */

void flogical_mesh_to_real(E,data,level)
     struct All_variables *E;
     float *data;
     int level;

{ int i,j,n1,n2;

  return;
}


void p_to_centres(E,PN,P,lev)
     struct All_variables *E;
     float **PN;
     double **P;
     int lev;

{  int p,element,node,j,m;
   double weight;

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(p=1;p<=E->lmesh.NEL[lev];p++)
      P[m][p] = 0.0;

   weight=1.0/((double)enodes[E->mesh.nsd]) ;

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(p=1;p<=E->lmesh.NEL[lev];p++)
      for(j=1;j<=enodes[E->mesh.nsd];j++)
        P[m][p] += PN[m][E->IEN[lev][m][p].node[j]] * weight;

   return;
   }


void v_to_intpts(E,VN,VE,lev)
  struct All_variables *E;
  float **VN,**VE;
  int lev;
  {

   int m,e,i,j,k;
   const int nsd=E->mesh.nsd;
   const int vpts=vpoints[nsd];
   const int ends=enodes[nsd];

 for (m=1;m<=E->sphere.caps_per_proc;m++)
   for(e=1;e<=E->lmesh.NEL[lev];e++)
     for(i=1;i<=vpts;i++)                 {
        VE[m][(e-1)*vpts + i] = 0.0;
        for(j=1;j<=ends;j++)
          VE[m][(e-1)*vpts + i] += VN[m][E->IEN[lev][m][e].node[j]]*E->N.vpt[GNVINDEX(j,i)];
        }

   return;
  }


void visc_to_intpts(E,VN,VE,lev)
   struct All_variables *E;
   float **VN,**VE;
   int lev;
   {

   int m,e,i,j,k;
   const int nsd=E->mesh.nsd;
   const int vpts=vpoints[nsd];
   const int ends=enodes[nsd];

 for (m=1;m<=E->sphere.caps_per_proc;m++)
   for(e=1;e<=E->lmesh.NEL[lev];e++)
     for(i=1;i<=vpts;i++) {
        VE[m][(e-1)*vpts + i] = 0.0;
	for(j=1;j<=ends;j++)
          VE[m][(e-1)*vpts + i] += log(VN[m][E->IEN[lev][m][e].node[j]]) *  E->N.vpt[GNVINDEX(j,i)];
        VE[m][(e-1)*vpts + i] = exp(VE[m][(e-1)*vpts + i]);
        }

  }


/* ==========================================================  */
/*  From Pan_problem_misc_functions.c                          */
/* =========================================================== */

double SIN_D(x)
     double x;
{
#if defined(__osf__)
  return sind(x);
#else
  return sin((x/180.0) * M_PI);
#endif

}

double COT_D(x)
     double x;
{
#if defined(__osf__)
  return cotd(x);
#else
  return tan(((90.0-x)/180.0) * M_PI);
#endif

}


/* non-runaway malloc */

void * Malloc1(bytes,file,line)
    int bytes;
    char *file;
    int line;
{
    void *ptr;

    ptr = malloc((size_t)bytes);
    if (ptr == (void *)NULL) {
	fprintf(stderr,"Memory: cannot allocate another %d bytes \n(line %d of file %s)\n",bytes,line,file);
	parallel_process_termination();
    }

    return(ptr);
}


/* returns the out of plane component of the cross product of
   the two vectors assuming that one is looking AGAINST the
   direction of the axis of D, anti-clockwise angles
   are positive (are you sure ?), and the axes are ordered 2,3 or 1,3 or 1,2 */


float cross2d(x11,x12,x21,x22,D)
    float x11,x12,x21,x22;
    int D;
{
  float temp;
   if(1==D)
       temp = ( x11*x22-x12*x21);
   if(2==D)
       temp = (-x11*x22+x12*x21);
   if(3==D)
       temp = ( x11*x22-x12*x21);

   return(temp);
}


/* Read in a file containing previous values of a field. The input in the parameter
   file for this should look like: `previous_name_file=string' and `previous_name_column=int'
   where `name' is substituted by the argument of the function.

   The file should have the standard CITCOM output format:
     # HEADER LINES etc
     index X Z Y ... field_value1 ...
     index X Z Y ... field_value2 ...
   where index is the node number, X Z Y are the coordinates and
   the field value is in the column specified by the abbr term in the function argument

   If the number of nodes OR the XZY coordinates for the node number (to within a small tolerance)
   are not in agreement with the existing mesh, the data is interpolated.

   */

int read_previous_field(E,field,name,abbr)
    struct All_variables *E;
    float **field;
    char *name, *abbr;
{
    int input_string();

    char discard[5001];
    char *token;
    char *filename;
    char *input_token;
    FILE *fp;
    int fnodesx,fnodesz,fnodesy;
    int i,j,column,found,m;

    float *X,*Z,*Y;

    filename=(char *)malloc(500*sizeof(char));
    input_token=(char *)malloc(1000*sizeof(char));

    /* Define field name, read parameter file to determine file name and column number */

    sprintf(input_token,"previous_%s_file",name);
    if(!input_string(input_token,filename,"initialize",E->parallel.me)) {
	fprintf(E->fp,"No previous %s information found in input file\n",name);fflush(E->fp);
	return(0);   /* if not found, take no further action, return zero */
    }


    fprintf(E->fp,"Previous %s information is in file %s\n",name,filename);fflush(E->fp);

    /* Try opening the file, fatal if this fails too */

    if((fp=fopen(filename,"r")) == NULL) {
	fprintf(E->fp,"Unable to open the required file `%s' (this is fatal)",filename);
	fflush(E->fp);

	parallel_process_termination();
    }


     /* Read header, get nodes xzy */

    fgets(discard,4999,fp);
    fgets(discard,4999,fp);
    i=sscanf(discard,"# NODESX=%d NODESZ=%d NODESY=%d",&fnodesx,&fnodesz,&fnodesy);
    if(i<3) {
	fprintf(E->fp,"File %s is not in the correct format\n",filename);fflush(E->fp);
	exit(1);
    }

    fgets(discard,4999,fp); /* largely irrelevant line */
    fgets(discard,4999,fp);

    /* this last line is the column headers, we need to search for the occurence of abbr to
       find out the column to be read in */

    if(strtok(discard,"|")==NULL) {
	fprintf(E->fp,"Unable to deciphre the columns in the input file");fflush(E->fp);
	exit(1);
    }

    found=0;
    column=1;

    while(found==0 && (token=strtok(NULL,"|")) != NULL) {
	if(strstr(token,abbr)!=0)
	    found=1;
	column++;
    }

    if(found) {
	fprintf(E->fp,"\t%s (%s) found in column %d\n",name,abbr,column);fflush(E->fp);
    }
    else {
	fprintf(E->fp,"\t%s (%s) not found in file: %s\n",name,abbr,filename);fflush(E->fp);
	exit(1);
    }



    /* Another fatal condition (not suitable for interpolation: */
    if(((3!= E->mesh.nsd) && (fnodesy !=1)) || ((3==E->mesh.nsd) && (1==fnodesy))) {
	fprintf(E->fp,"Input data for file `%s'  is of inappropriate dimension (not %dD)\n",filename,E->mesh.nsd);fflush(E->fp);
	exit(1);
    }

    if(fnodesx != E->lmesh.nox || fnodesz != E->lmesh.noz || fnodesy != E->lmesh.noy) {
       fprintf(stderr,"wrong dimension in the input temperature file!!!!\n");
       exit(1);
       }

    X=(float *)malloc((2+fnodesx*fnodesz*fnodesy)*sizeof(float));
    Z=(float *)malloc((2+fnodesx*fnodesz*fnodesy)*sizeof(float));
    Y=(float *)malloc((2+fnodesx*fnodesz*fnodesy)*sizeof(float));

   /* Format for reading the input file (including coordinates) */

    sprintf(input_token," %%d %%e %%e %%e");
    for(i=5;i<column;i++)
	strcat(input_token," %*f");
    strcat(input_token," %f");


    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(i=1;i<=fnodesx*fnodesz*fnodesy;i++) {
	fgets(discard,4999,fp);
	sscanf(discard,input_token,&j,&(X[i]),&(Z[i]),&(Y[i]),&field[m][i]);
        }
    /* check consistency & need for interpolation */

    fclose(fp);


    free((void *)X);
    free((void *)Z);
    free((void *)Y);
    free((void *)filename);
    free((void *)input_token);

    return(1);
}



/* ==========================================================  */
/*  From General_matrix_functions.c                            */
/* =========================================================== */

/*=====================================================================
  Variable dimension matrix allocation  function from numerical recipes
  Note: ANSII consistency requires some additional features !
  =====================================================================  */

double **dmatrix(nrl,nrh,ncl,nch)
     int nrl,nrh,ncl,nch;
{
  int i,nrow = nrh-nrl+1,ncol=nch-ncl+1;
  double **m;

  /* allocate pointer to rows  */
  m=(double **) malloc((nrow+1)* sizeof(double *));
  m+=1;
  m-= nrl;

  /*  allocate rows and set the pointers accordingly   */
  m[nrl] = (double *) malloc((nrow*ncol+1)* sizeof(double));
  m[nrl] += 1;
  m[nrl] -= ncl;

  for(i=nrl+1;i<=nrh;i++)
     m[i] = m[i-1] + ncol;

  return(m);		}


float **fmatrix(nrl,nrh,ncl,nch)
     int nrl,nrh,ncl,nch;
{
  int i,nrow = nrh-nrl+1,ncol=nch-ncl+1;
  float **m;

  /* allocate pointer to rows  */
  m=(float **) malloc((unsigned)((nrow+1)* sizeof(float *)));
  m+=1;
  m-= nrl;

  /*  allocate rows and set the pointers accordingly   */
  m[nrl] = (float *) malloc((unsigned)((nrow*ncol+1)* sizeof(float)));
  m[nrl] += 1;
  m[nrl] -= ncl;

  for(i=nrl+1;i<=nrh;i++)
     m[i] = m[i-1] + ncol;

  return(m);		}


void dfree_matrix(m,nrl,nrh,ncl,nch)
     double **m;
     int nrl,nrh,ncl,nch;
{
  int i;
  for(i=nrh;i>=nrl;i--)
    free((void *)(m[i] + ncl));
  free((void *) (m+nrl));
  return;
}

void ffree_matrix(m,nrl,nrh,ncl,nch)
     float **m;
     int nrl,nrh,ncl,nch;
{
  int i;
  for(i=nrh;i>=nrl;i--)
    free((void *)(m[i] + ncl));
  free((void *) (m+nrl));
  return;
}

/*=============================================================
  Functions to allocate/remove space for variable sized vector.
  =============================================================  */

double *dvector(nl,nh)
     int nl,nh;
{
  double *v;
  v=(double *) malloc((unsigned) ( nh - nl +1)* sizeof(double));
  return( v-nl );  }

float *fvector(nl,nh)
     int nl,nh;
{
  float *v;
  v=(float *) malloc((unsigned) ( nh - nl +1)* sizeof(float));
  return( v-nl );  }

void dfree_vector(v,nl,nh)
     double *v;
     int nl,nh;
{
  free((char*) (v+nl));	}

void ffree_vector(v,nl,nh)
     float *v;
     int nl,nh;
{
  free((char*) (v+nl));	}

int *sivector(nl,nh)
     int nl,nh;
{
  int *v;
  v=(int*) malloc((unsigned)(nh-nl +1) * sizeof(int));
  return (v-nl);
}

void sifree_vector(v,nl,nh)
     int *v;
     int nl,nh;
{ free((char *) (v+nl));    }



void dvcopy(E,A,B,a,b)
     struct All_variables *E;
     double **A,**B;
     int a,b;

{   int i,m;

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=a;i<=b;i++)
      A[m][i] = B[m][i];

    return; }

void vcopy(A,B,a,b)
     float *A,*B;
     int a,b;

{   int i;

    for(i=a;i<=b;i++)
      A[i] = B[i];

    return; }



/* =====================================*/
 double sphere_h(l,m,t,f,ic)
 int l,m,ic;
 double t,f;
 {

 double plgndr_a(),sphere_hamonics;

 sphere_hamonics = 0.0;
 if (ic==0)
    sphere_hamonics = cos(m*f)*plgndr_a(l,m,t);
 else if (m)
    sphere_hamonics = sin(m*f)*plgndr_a(l,m,t);

 return sphere_hamonics;
 }

/* =====================================*/
 double plgndr_a(l,m,t)
 int l,m;
 double t;
 {

  int i,ll;
  double x,fact,pll,pmm,pmmp1,somx2,plgndr;
  const double two=2.0;
  const double one=1.0;

  x = cos(t);
  pmm=one;
  if(m>0) {
    somx2=sqrt((one-x)*(one+x));
    fact = one;
    for (i=1;i<=m;i++)   {
      pmm = -pmm*fact*somx2;
      fact = fact + two;
      }
    }

  if (l==m)
     plgndr = pmm;
  else  {
     pmmp1 = x*(2*m+1)*pmm;
     if(l==m+1)
       plgndr = pmmp1;
     else   {
       for (ll=m+2;ll<=l;ll++)  {
         pll = (x*(2*ll-1)*pmmp1-(ll+m-1)*pmm)/(ll-m);
         pmm = pmmp1;
         pmmp1 = pll;
         }
       plgndr = pll;
       }
     }

 return plgndr;
 }

 float area_of_4node(x1,y1,x2,y2,x3,y3,x4,y4)
 float x1,y1,x2,y2,x3,y3,x4,y4;

 {
 float area;

 area = fabs(0.5*(x1*(y2-y4)+x2*(y4-y1)+x4*(y1-y2)))
      + fabs(0.5*(x2*(y3-y4)+x3*(y4-y2)+x4*(y2-y3)));

 return area;
 }

void print_elt_k(E,a)
     struct All_variables *E;
     double a[24*24];

{ int l,ll,n;

  printf("elt k is ...\n");


  n = loc_mat_size[E->mesh.nsd];

  for(l=0;l<n;l++)
    { fprintf(stderr,"\n");fflush(stderr);
      for(ll=0;ll<n;ll++)
	{ fprintf(stderr,"%s%.3e ",a[ll*n+l] >= 0.0 ? "+" : "",a[ll*n+l]);
	  fflush(stderr);
	}
    }
  fprintf(stderr,"\n"); fflush(stderr);

  return; }


 /* ===================================  */
  double sqrt_multis(jj,ii)
  int ii,jj;
 {
  int i;
  double sqrt_multisa;

  sqrt_multisa = 1.0;
  if(jj>ii)
    for (i=jj;i>ii;i--)
      sqrt_multisa *= 1.0/sqrt((double)i);

  return sqrt_multisa;
  }

 /* ===================================  */
  double multis(ii)
  int ii;
 {
  int i;
  double multisa;

  multisa = 1.0;
  if (ii)
    for (i=2;i<=ii;i++)
      multisa *= (double)i;

  return multisa;
  }


 /* ===================================  */
 int int_multis(ii)
 int ii;
 {
 int i,multisa;

 multisa = 1;
 if (ii)
   for (i=2;i<=ii;i++)
     multisa *= i;

 return multisa;
 }


void jacobi(E,d0,F,Ad,acc,cycles,level,guess)
     struct All_variables *E;
     double **d0;
     double **F,**Ad;
     double acc;
     int *cycles;
     int level;
     int guess;
{

    int count,i,j,k,l,m,ns,steps;
    int *C;
    int eqn1,eqn2,eqn3,gneq;

    void n_assemble_del2_u();

    double sum1,sum2,sum3,residual,global_vdot(),U1,U2,U3;

    double *r1[NCS];

    higher_precision *B1,*B2,*B3;


    const int dims=E->mesh.nsd;
    const int ends=enodes[dims];
    const int n=loc_mat_size[E->mesh.nsd];
    const int neq=E->lmesh.NEQ[level];
    const int num_nodes=E->lmesh.NNO[level];
    const int nox=E->lmesh.NOX[level];
    const int noz=E->lmesh.NOY[level];
    const int noy=E->lmesh.NOZ[level];
    const int max_eqn=14*dims;

    gneq = E->mesh.NEQ[level];

    steps=*cycles;

    for (m=1;m<=E->sphere.caps_per_proc;m++)
      r1[m] = (double *)malloc(E->lmesh.neq*sizeof(double));

    if(guess) {
      for (m=1;m<=E->sphere.caps_per_proc;m++)
          d0[m][neq]=0.0;
      n_assemble_del2_u(E,d0,Ad,level,1);
    }
    else
      for (m=1;m<=E->sphere.caps_per_proc;m++)
	for(i=0;i<=neq;i++) {
	    d0[m][i]=Ad[m][i]=0.0;
	}

    for (m=1;m<=E->sphere.caps_per_proc;m++)
      for(i=0;i<neq;i++)
        r1[m][i]=F[m][i]-Ad[m][i];


    count = 0;

   while (count < steps)   {
      for (m=1;m<=E->sphere.caps_per_proc;m++)
 	for(i=1;i<=E->lmesh.NNO[level];i++)  {
	    eqn1=E->ID[level][m][i].doff[1];
	    eqn2=E->ID[level][m][i].doff[2];
	    eqn3=E->ID[level][m][i].doff[3];
            d0[m][eqn1] += r1[m][eqn1]*E->BI[level][m][eqn1];
            d0[m][eqn2] += r1[m][eqn2]*E->BI[level][m][eqn2];
            d0[m][eqn3] += r1[m][eqn3]*E->BI[level][m][eqn3];
            }

      for (m=1;m<=E->sphere.caps_per_proc;m++)
	for(i=0;i<=neq;i++)
	    Ad[m][i]=0.0;

      for (m=1;m<=E->sphere.caps_per_proc;m++)
 	for(i=1;i<=E->lmesh.NNO[level];i++)  {
	    eqn1=E->ID[level][m][i].doff[1];
	    eqn2=E->ID[level][m][i].doff[2];
	    eqn3=E->ID[level][m][i].doff[3];
	    U1 = d0[m][eqn1];
	    U2 = d0[m][eqn2];
	    U3 = d0[m][eqn3];

            C=E->Node_map[level][m]+(i-1)*max_eqn;
	    B1=E->Eqn_k1[level][m]+(i-1)*max_eqn;
	    B2=E->Eqn_k2[level][m]+(i-1)*max_eqn;
 	    B3=E->Eqn_k3[level][m]+(i-1)*max_eqn;

            for(j=3;j<max_eqn;j++)  {
               Ad[m][eqn1] += B1[j]*d0[m][C[j]];
               Ad[m][eqn2] += B2[j]*d0[m][C[j]];
               Ad[m][eqn3] += B3[j]*d0[m][C[j]];
               }

	    for(j=0;j<max_eqn;j++) {
		    Ad[m][C[j]]  += B1[j]*U1 +  B2[j]*U2 +  B3[j]*U3;
		}
	    }       /* end for i and m */

      (E->solver.exchange_id_d)(E, Ad, level);

      for (m=1;m<=E->sphere.caps_per_proc;m++)
	for(i=0;i<neq;i++)
	    r1[m][i] = F[m][i] - Ad[m][i];

   /*   residual = sqrt(global_vdot(E,r1,r1,level))/gneq;

   if(E->parallel.me==0)fprintf(stderr,"residuall =%.5e for %d\n",residual,count);
*/	count++;
    }

   *cycles=count;

    for (m=1;m<=E->sphere.caps_per_proc;m++)
      free((double*) r1[m]);

    return;

    }


/* version */
/* $Id$ */

/* End of file  */
