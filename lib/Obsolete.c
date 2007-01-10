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

    E->sphere.con = (double *)malloc((E->sphere.hindice+3)*sizeof(double));
    for (ll=0;ll<=E->output.llmax;ll++)
	for (mm=0;mm<=ll;mm++)   {
	    E->sphere.con[E->sphere.hindex[ll][mm]] =
		sqrt( (2.0-((mm==0)?1.0:0.0))*(2*ll+1)/(4.0*M_PI) )
		*sqrt_multis(ll+mm,ll-mm);  /* which is sqrt((ll-mm)!/(ll+mm)!) */
	}

    E->sphere.tablenplm   = (double **) malloc((E->sphere.nox+1)
					       *sizeof(double*));
    for (i=1;i<=E->sphere.nox;i++)
	E->sphere.tablenplm[i]= (double *)malloc((E->sphere.hindice+3)
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

/* version */
/* $Id$ */

/* End of file  */
