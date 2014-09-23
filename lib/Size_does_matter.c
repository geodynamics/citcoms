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
#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"

#ifdef ALLOW_ELLIPTICAL
double theta_g(double , struct All_variables *);
#endif

void twiddle_thumbs(struct All_variables *E)
{
}

/*   ======================================================================
     ======================================================================  */

static void form_rtf_bc(int k, double x[4],
                        double rtf[4][9], double bc[4][4])
{
    double myatan();

    rtf[3][k] = 1.0/sqrt(x[1]*x[1]+x[2]*x[2]+x[3]*x[3]); /* 1/r */
    rtf[1][k] = acos(x[3]*rtf[3][k]); /* theta */
    rtf[2][k] = myatan(x[2],x[1]); /* phi */

    bc[1][1] = x[3]*cos(rtf[2][k]); /* theta */
    bc[1][2] = x[3]*sin(rtf[2][k]);
    bc[1][3] = -sin(rtf[1][k])/rtf[3][k];
    bc[2][1] = -x[2];		/* phi basis vector */
    bc[2][2] = x[1];
    bc[2][3] = 0.0;
    bc[3][1] = x[1]*rtf[3][k];	/*  */
    bc[3][2] = x[2]*rtf[3][k];
    bc[3][3] = x[3]*rtf[3][k];
}


static void get_global_shape_fn_sph(struct All_variables *E,
                                    int lev, int el)
{
    int i,j,k,d,e;
    double jacobian;
    double determinant();
    double cofactor(),myatan();
    void   form_rtf_bc();

    struct Shape_function_dx LGNx;

    double dxda[4][4], cof[4][4], x[4], rtf[4][9], bc[4][4];

    const int dims = E->mesh.nsd;
    const int ends = ENODES3D;
    const int vpts = VPOINTS3D;
    const int ppts = PPOINTS3D;


    for(k=1;k<=vpts;k++) {       /* all of the vpoints */
        for(d=1;d<=dims;d++)  {
            x[d]=0.0;
            for(e=1;e<=dims;e++)
                dxda[d][e]=0.0;
        }

        for(d=1;d<=dims;d++)
            for(i=1;i<=ends;i++)
                x[d] += E->X[lev][CPPR][d][E->IEN[lev][CPPR][el].node[i]]
                    * E->N.vpt[GNVINDEX(i,k)];

        for(d=1;d<=dims;d++)
            for(e=1;e<=dims;e++)
                for(i=1;i<=ends;i++)
                    dxda[d][e] += E->X[lev][CPPR][e][E->IEN[lev][CPPR][el].node[i]]
                        * E->Nx.vpt[GNVXINDEX(d-1,i,k)];

        jacobian = determinant(dxda, E->mesh.nsd);
        E->GDA[lev][CPPR][el].vpt[k] = jacobian;

        for(d=1;d<=dims;d++)
            for(e=1;e<=dims;e++)
                cof[d][e]=cofactor(dxda,d,e,dims);

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
                E->GNX[lev][CPPR][el].vpt[GNVXINDEX(d-1,j,k)]
                    = bc[d][1]*LGNx.vpt[GNVXINDEX(0,j,k)]
                    + bc[d][2]*LGNx.vpt[GNVXINDEX(1,j,k)]
                    + bc[d][3]*LGNx.vpt[GNVXINDEX(2,j,k)];
            }
    }     /* end for k */

    for(k=1;k<=ppts;k++) {   /* all of the ppoints */
        for(d=1;d<=dims;d++) {
            x[d]=0.0;
            for(e=1;e<=dims;e++)
                dxda[d][e]=0.0;
        }

        for(d=1;d<=dims;d++)
            for(i=1;i<=ends;i++)
                x[d] += E->X[lev][CPPR][d][E->IEN[lev][CPPR][el].node[i]]
                    * E->N.ppt[GNPINDEX(i,k)];

        for(d=1;d<=dims;d++)
            for(e=1;e<=dims;e++)
                for(i=1;i<=ends;i++)
                    dxda[d][e] += E->X[lev][CPPR][e][E->IEN[lev][CPPR][el].node[i]]
                        * E->Nx.ppt[GNPXINDEX(d-1,i,k)];

        jacobian = determinant(dxda,E->mesh.nsd);
        E->GDA[lev][CPPR][el].ppt[k] = jacobian;

        for(d=1;d<=dims;d++)
            for(e=1;e<=dims;e++)
                cof[d][e]=cofactor(dxda,d,e,E->mesh.nsd);

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
                E->GNX[lev][CPPR][el].ppt[GNPXINDEX(d-1,j,k)]
                    = bc[d][1]*LGNx.ppt[GNPXINDEX(0,j,k)]
                    + bc[d][2]*LGNx.ppt[GNPXINDEX(1,j,k)]
                    + bc[d][3]*LGNx.ppt[GNPXINDEX(2,j,k)];
            }

    }              /* end for k int */
}


void construct_shape_function_derivatives(struct All_variables *E)
{
    int m, lev, el;

    for(lev=E->mesh.levmax; lev>=E->mesh.levmin; lev--)
        for(el=1; el<=E->lmesh.NEL[lev]; el++)
            get_global_shape_fn_sph(E, lev, el);
}


/* 

gets r,theta,phi coordinates at the integration points

 */
void get_rtf_at_vpts(struct All_variables *E, int lev, int el, double rtf[4][9])
{
    int i, k, d;
    double x[4];

    double myatan();

    const int dims = E->mesh.nsd;
    const int ends = ENODES3D;
    const int vpts = VPOINTS3D;

    for(k=1;k<=vpts;k++) {       /* all of the vpoints */
        for(d=1;d<=dims;d++)
            x[d]=0.0;

        for(d=1;d<=dims;d++)
            for(i=1;i<=ends;i++)
                x[d] += E->X[lev][CPPR][d][E->IEN[lev][CPPR][el].node[i]]
                    * E->N.vpt[GNVINDEX(i,k)];

        rtf[3][k] = 1.0/sqrt(x[1]*x[1]+x[2]*x[2]+x[3]*x[3]); /* 1/r */
        rtf[1][k] = acos(x[3]*rtf[3][k]); /* theta */
        rtf[2][k] = myatan(x[2],x[1]); /* phi */
    }
}


void get_rtf_at_ppts(struct All_variables *E, int lev, int el, double rtf[4][9])
{
    int i, k, d;
    double x[4];

    double myatan();

    const int dims = E->mesh.nsd;
    const int ends = ENODES3D;
    const int ppts = PPOINTS3D;

    for(k=1;k<=ppts;k++) {   /* all of the ppoints */
        for(d=1;d<=dims;d++)
            x[d]=0.0;

        for(d=1;d<=dims;d++)
            for(i=1;i<=ends;i++)
                x[d] += E->X[lev][CPPR][d][E->IEN[lev][CPPR][el].node[i]]
                    * E->N.ppt[GNPINDEX(i,k)];

        rtf[3][k] = 1.0/sqrt(x[1]*x[1]+x[2]*x[2]+x[3]*x[3]);
        rtf[1][k] = acos(x[3]*rtf[3][k]);
        rtf[2][k] = myatan(x[2],x[1]);
    }
}


void get_side_x_cart(struct All_variables *E, double xx[4][5], int el, int side)
{
  double to,fo,dxdy[4][4];
  int i, node, s;
  const int oned = onedvpoints[E->mesh.nsd];

#ifdef ALLOW_ELLIPTICAL
  to = theta_g(E->eco[el].centre[1],E);
#else
  to = E->eco[el].centre[1];	
#endif

  fo = E->eco[el].centre[2];

  dxdy[1][1] = cos(to)*cos(fo);
  dxdy[1][2] = cos(to)*sin(fo);
  dxdy[1][3] = -sin(to);
  dxdy[2][1] = -sin(fo);
  dxdy[2][2] = cos(fo);
  dxdy[2][3] = 0.0;
  dxdy[3][1] = sin(to)*cos(fo);
  dxdy[3][2] = sin(to)*sin(fo);
  dxdy[3][3] = cos(to);

  for(i=1;i<=oned;i++) {     /* nodes */
    s = sidenodes[side][i];
    node = E->ien[el].node[s];
    xx[1][i] = E->x[CPPR][1][node]*dxdy[1][1]
             + E->x[CPPR][2][node]*dxdy[1][2]
             + E->x[CPPR][3][node]*dxdy[1][3];
    xx[2][i] = E->x[CPPR][1][node]*dxdy[2][1]
             + E->x[CPPR][2][node]*dxdy[2][2]
             + E->x[CPPR][3][node]*dxdy[2][3];
    xx[3][i] = E->x[CPPR][1][node]*dxdy[3][1]
             + E->x[CPPR][2][node]*dxdy[3][2]
             + E->x[CPPR][3][node]*dxdy[3][3];
  }
}


/*   ======================================================================
     ======================================================================  */
void construct_surf_det (E)
     struct All_variables *E;
{

  int m,i,k,d,e,es,el;

  double jacobian;
  double determinant();
  double cofactor();

  const int oned = onedvpoints[E->mesh.nsd];

  double xx[4][5], dxda[4][4], r2;

    for(k=1;k<=oned;k++)    { /* all of the vpoints*/
      E->surf_det[CPPR][k] = (double *)malloc((1+E->lmesh.snel)*sizeof(double));
    }

  r2 = 1.0 / (E->sx[3][E->lmesh.elz+1] * E->sx[3][E->lmesh.elz+1]);

  for (es=1;es<=E->lmesh.snel;es++)   {
    el = es * E->lmesh.elz;
    get_side_x_cart(E, xx, el, SIDE_TOP);

    for(k=1;k<=oned;k++)    { /* all of the vpoints*/
      for(d=1;d<=E->mesh.nsd-1;d++)
        for(e=1;e<=E->mesh.nsd-1;e++)
            dxda[d][e]=0.0;

      for(i=1;i<=oned;i++)      /* nodes */
        for(d=1;d<=E->mesh.nsd-1;d++)
          for(e=1;e<=E->mesh.nsd-1;e++)
             dxda[d][e] += xx[e][i]*E->Mx.vpt[GMVXINDEX(d-1,i,k)];

      jacobian = determinant(dxda,E->mesh.nsd-1);

      /* scale the jacobian so that it is defined on a unit sphere */
      E->surf_det[CPPR][k][es] = jacobian * r2;
      }
    }
}



/*   ======================================================================
     surface (6 sides) determinant of boundary element
     ======================================================================  */
void construct_bdry_det(struct All_variables *E)
{

  int m,i,k,d,e,es,el,side;

  double jacobian;
  double determinant();
  double cofactor();

  const int oned = onedvpoints[E->mesh.nsd];

  double xx[4][5],dxda[4][4];

    for (side=SIDE_BEGIN; side<=SIDE_END; side++)
      for(d=1; d<=oned; d++)
	E->boundary.det[CPPR][side][d] = (double *)malloc((1+E->boundary.nel)*sizeof(double));

    for (es=1;es<=E->boundary.nel;es++) {
      el = E->boundary.element[CPPR][es];

      for (side=SIDE_BEGIN; side<=SIDE_END; side++) {
	get_side_x_cart(E, xx, el, side);

	for(k=1;k<=oned;k++) { /* all of the vpoints*/

	  for(d=1;d<=E->mesh.nsd-1;d++)
	    for(e=1;e<=E->mesh.nsd-1;e++)
	      dxda[d][e]=0.0;

	  for(i=1;i<=oned;i++) /* nodes */
	    for(d=1;d<=E->mesh.nsd-1;d++)
	      for(e=1;e<=E->mesh.nsd-1;e++)
		dxda[d][e] += xx[sidedim[side][e]][i]*E->Mx.vpt[GMVXINDEX(d-1,i,k)];

	  jacobian = determinant(dxda,E->mesh.nsd-1);
	  E->boundary.det[CPPR][side][k][es] = jacobian;
	}

      }
    }
}



/*   ======================================================================
     ======================================================================  */
void get_global_1d_shape_fn(E,el,GM,dGammax,top)
     struct All_variables *E;
     int el,top;
     struct Shape_function1 *GM;
     struct Shape_function1_dA *dGammax;
{
  int ii,i,k,d,e;

  double jacobian;
  double determinant();

  const int oned = onedvpoints[E->mesh.nsd];

  double xx[4][5],dxda[4][4];

  for (ii=0;ii<=top;ii++)   {   /* ii=0 for bottom and ii=1 for top */

    get_side_x_cart(E, xx, el, ii+1);

    for(k=1;k<=oned;k++)    { /* all of the vpoints*/
      for(d=1;d<=E->mesh.nsd-1;d++)
	for(e=1;e<=E->mesh.nsd-1;e++)
	  dxda[d][e]=0.0;

      for(i=1;i<=oned;i++)      /* nodes */
	for(d=1;d<=E->mesh.nsd-1;d++)
	  for(e=1;e<=E->mesh.nsd-1;e++)
	    dxda[d][e] += xx[e][i]*E->Mx.vpt[GMVXINDEX(d-1,i,k)];

      jacobian = determinant(dxda,E->mesh.nsd-1);
      dGammax->vpt[GMVGAMMA(ii,k)] = jacobian;
    }
  }
}

/*   ======================================================================
     ======================================================================  */
void get_global_1d_shape_fn_L(E,el,GM,dGammax,top)
     struct All_variables *E;
     int el,top;
     struct Shape_function1 *GM;
     struct Shape_function1_dA *dGammax;
{
    int ii,i,k,d,e,node;

    double jacobian;
    double determinant();

    const int oned = onedvpoints[E->mesh.nsd];

    double to,fo,xx[4][5],dxdy[4][4],dxda[4][4],cof[4][4];

#ifdef ALLOW_ELLIPTICAL
    to = theta_g(E->eco[el].centre[1],E);
#else
    to = E->eco[el].centre[1]; 
#endif
    fo = E->eco[el].centre[2];

    dxdy[1][1] = cos(to)*cos(fo);
    dxdy[1][2] = cos(to)*sin(fo);
    dxdy[1][3] = -sin(to);
    dxdy[2][1] = -sin(fo);
    dxdy[2][2] = cos(fo);
    dxdy[2][3] = 0.0;
    dxdy[3][1] = sin(to)*cos(fo);
    dxdy[3][2] = sin(to)*sin(fo);
    dxdy[3][3] = cos(to);

    for (ii=0;ii<=top;ii++)   {   /* ii=0 for bottom and ii=1 for top */

        for(i=1;i<=oned;i++) {     /* nodes */
            e = i+ii*oned;
            node = E->ien[el].node[e];
            xx[1][i] = E->x[CPPR][1][node]*dxdy[1][1]
                + E->x[CPPR][2][node]*dxdy[1][2]
                + E->x[CPPR][3][node]*dxdy[1][3];
            xx[2][i] = E->x[CPPR][1][node]*dxdy[2][1]
                + E->x[CPPR][2][node]*dxdy[2][2]
                + E->x[CPPR][3][node]*dxdy[2][3];
            xx[3][i] = E->x[CPPR][1][node]*dxdy[3][1]
                + E->x[CPPR][2][node]*dxdy[3][2]
                + E->x[CPPR][3][node]*dxdy[3][3];
        }

        for(k=1;k<=oned;k++)    { /* all of the vpoints*/

            for(d=1;d<=E->mesh.nsd-1;d++)
                for(e=1;e<=E->mesh.nsd-1;e++)
                    dxda[d][e]=0.0;

            for(i=1;i<=oned;i++)      /* nodes */
                for(d=1;d<=E->mesh.nsd-1;d++)
                    for(e=1;e<=E->mesh.nsd-1;e++)
                        dxda[d][e] += xx[e][i]*E->Lx.vpt[GMVXINDEX(d-1,i,k)];

            jacobian = determinant(dxda,E->mesh.nsd-1);
            dGammax->vpt[GMVGAMMA(ii,k)] = jacobian;
        }
    }
}

/*   ======================================================================
     For calculating pressure boundary term --- Choi, 11/13/02
     ======================================================================  */
void get_global_side_1d_shape_fn(E,el,GM,GMx,dGamma,side)
     struct All_variables *E;
     int el,side;
     struct Shape_function1 *GM;
     struct Shape_function1_dx *GMx;
     struct Shape_function_side_dA *dGamma;
{
  int i,k,d,e;

  double jacobian;
  double determinant();

  const int oned = onedvpoints[E->mesh.nsd];
  double xx[4][5],dxda[4][4];

  get_side_x_cart(E, xx, el, side);

  for(k=1;k<=oned;k++)    {

    for(d=1;d<=E->mesh.nsd-1;d++)
      for(e=1;e<=E->mesh.nsd-1;e++)
	dxda[d][e]=0.0;

    for(i=1;i<=oned;i++) {
      for(d=1;d<=E->mesh.nsd-1;d++)
	for(e=1;e<=E->mesh.nsd-1;e++) {
	  dxda[d][e] += xx[sidedim[side][e]][i]*E->Mx.vpt[GMVXINDEX(d-1,i,k)];
	}
    }

    jacobian = determinant(dxda,E->mesh.nsd-1);
    dGamma->vpt[k] = jacobian;
  }
}


/* ====================================================   */

void construct_c3x3matrix_el (struct All_variables *E,int el,struct CC *cc,
			      struct CCX *ccx,int lev,int pressure)
{
  int a,i,j,k,d,lnode;
  double cofactor(),myatan();
  double x[4],u[4][4],ux[3][4][4],ua[4][4];
  double costt,cosff,sintt,sinff,rr,tt,ff;

  const int dims=E->mesh.nsd;
  const int ends=enodes[dims];
  const int vpts=vpoints[dims];
  const int ppts=ppoints[dims];

  if (pressure==0)           {
    for(k=1;k<=vpts;k++)           {       /* all of the vpoints */
      for(d=1;d<=dims;d++)
          x[d]=0.0;

      for(d=1;d<=dims;d++)
          for(a=1;a<=ends;a++)
            x[d] += E->X[lev][CPPR][d][E->IEN[lev][CPPR][el].node[a]]
                   *E->N.vpt[GNVINDEX(a,k)];

      rr = sqrt(x[1]*x[1]+x[2]*x[2]+x[3]*x[3]);

      ff = myatan(x[2],x[1]);
#ifdef ALLOW_ELLIPTICAL
      tt = theta_g(acos(x[3]/rr),E);
#else
      tt = acos(x[3]/rr);
#endif
      costt = cos(tt);		
      cosff = cos(ff);
      sintt = sin(tt);
      sinff = sin(ff);

      u[1][1] = costt*cosff; u[1][2] = costt*sinff;  u[1][3] =-sintt;
      u[2][1] =-sinff;       u[2][2] = cosff;        u[2][3] = 0.0;
      u[3][1] = sintt*cosff; u[3][2] = sintt*sinff;  u[3][3] = costt;

      ux[1][1][1] =-sintt*cosff;  ux[1][1][2] =-sintt*sinff;  ux[1][1][3] =-costt;
      ux[2][1][1] =-costt*sinff;  ux[2][1][2] = costt*cosff;  ux[2][1][3] =0.0;
      ux[1][2][1] =0.0;           ux[1][2][2] = 0.0;          ux[1][2][3] =0.0;
      ux[2][2][1] =-cosff;        ux[2][2][2] =-sinff;        ux[2][2][3] =0.0;
      ux[1][3][1] = costt*cosff;  ux[1][3][2] = costt*sinff;  ux[1][3][3] =-sintt;
      ux[2][3][1] =-sintt*sinff;  ux[2][3][2] = sintt*cosff;  ux[2][3][3] =0.0;

      for(a=1;a<=ends;a++)   {
          tt = E->SX[lev][CPPR][1][E->IEN[lev][CPPR][el].node[a]];
          ff = E->SX[lev][CPPR][2][E->IEN[lev][CPPR][el].node[a]];
          costt = cos(tt);
          cosff = cos(ff);
          sintt = sin(tt);
          sinff = sin(ff);

          ua[1][1] = costt*cosff; ua[1][2] = costt*sinff;  ua[1][3] =-sintt;
          ua[2][1] =-sinff;       ua[2][2] = cosff;        ua[2][3] = 0.0;
          ua[3][1] = sintt*cosff; ua[3][2] = sintt*sinff;  ua[3][3] = costt;

          for (i=1;i<=dims;i++)
            for (j=1;j<=dims;j++)   {
              cc->vpt[BVINDEX(i,j,a,k)] =
                    ua[j][1]*u[i][1]+ua[j][2]*u[i][2]+ua[j][3]*u[i][3];
              ccx->vpt[BVXINDEX(i,j,1,a,k)] =
                    ua[j][1]*ux[1][i][1]+ua[j][2]*ux[1][i][2]+ua[j][3]*ux[1][i][3];
              ccx->vpt[BVXINDEX(i,j,2,a,k)] =
                    ua[j][1]*ux[2][i][1]+ua[j][2]*ux[2][i][2]+ua[j][3]*ux[2][i][3];
              }
          }      /* end for local node */

        }        /* end for int points */
     }        /* end if */

   else if (pressure)  {

      for(k=1;k<=ppts;k++)           {       /* all of the ppoints */
        for(d=1;d<=dims;d++)
          x[d]=0.0;

        for(d=1;d<=dims;d++)
          for(a=1;a<=ends;a++)
            x[d] += E->X[lev][CPPR][d][E->IEN[lev][CPPR][el].node[a]]
                   *E->N.ppt[GNPINDEX(a,k)];

        rr = sqrt(x[1]*x[1]+x[2]*x[2]+x[3]*x[3]);

        ff = myatan(x[2],x[1]);
#ifdef ALLOW_ELLIPTICAL
	tt = theta_g(acos(x[3]/rr),E);
#else
        tt = acos(x[3]/rr);
#endif
        costt = cos(tt);	
        cosff = cos(ff);
        sintt = sin(tt);
        sinff = sin(ff);

        u[1][1] = costt*cosff; u[1][2] = costt*sinff;  u[1][3] =-sintt;
        u[2][1] =-sinff;       u[2][2] = cosff;        u[2][3] = 0.0;
        u[3][1] = sintt*cosff; u[3][2] = sintt*sinff;  u[3][3] = costt;

        ux[1][1][1] =-sintt*cosff;  ux[1][1][2] =-sintt*sinff;  ux[1][1][3] =-costt;
        ux[2][1][1] =-costt*sinff;  ux[2][1][2] = costt*cosff;  ux[2][1][3] =0.0;
        ux[1][2][1] =0.0;           ux[1][2][2] = 0.0;          ux[1][2][3] =0.0;
        ux[2][2][1] =-cosff;        ux[2][2][2] =-sinff;        ux[2][2][3] =0.0;
        ux[1][3][1] = costt*cosff;  ux[1][3][2] = costt*sinff;  ux[1][3][3] =-sintt;
        ux[2][3][1] =-sintt*sinff;  ux[2][3][2] = sintt*cosff;  ux[2][3][3] =0.0;

        for(a=1;a<=ends;a++)   {

	  lnode = E->IEN[lev][CPPR][el].node[a];
	  sintt = E->SinCos[lev][CPPR][0][lnode];
	  sinff = E->SinCos[lev][CPPR][1][lnode];
	  costt = E->SinCos[lev][CPPR][2][lnode];
	  cosff = E->SinCos[lev][CPPR][3][lnode];

          ua[1][1] = costt*cosff; ua[1][2] = costt*sinff;  ua[1][3] =-sintt;
          ua[2][1] =-sinff;       ua[2][2] = cosff;        ua[2][3] = 0.0;
          ua[3][1] = sintt*cosff; ua[3][2] = sintt*sinff;  ua[3][3] = costt;

          for (i=1;i<=dims;i++)
            for (j=1;j<=dims;j++)   {
              cc->ppt[BPINDEX(i,j,a,k)] =
                    ua[j][1]*u[i][1]+ua[j][2]*u[i][2]+ua[j][3]*u[i][3];
              ccx->ppt[BPXINDEX(i,j,1,a,k)] =
                    ua[j][1]*ux[1][i][1]+ua[j][2]*ux[1][i][2]+ua[j][3]*ux[1][i][3];
              ccx->ppt[BPXINDEX(i,j,2,a,k)] =
                    ua[j][1]*ux[2][i][1]+ua[j][2]*ux[2][i][2]+ua[j][3]*ux[2][i][3];
              }

          }      /* end for local node */

        }        /* end for int points */


      }         /* end if pressure  */
}


void construct_side_c3x3matrix_el(struct All_variables *E,int el,
				  struct CC *cc,struct CCX *ccx,
				  int lev,int pressure,int side)
{
  int a,aa,i,j,k,d,lnode;
  double cofactor(),myatan();
  double x[4],u[4][4],ux[3][4][4],ua[4][4];
  double costt,cosff,sintt,sinff,rr,tt,ff;

  const int dims=E->mesh.nsd;
  const int ends=enodes[dims-1];
  const int vpts=onedvpoints[dims];
  const int ppts=ppoints[dims];

  if(pressure==0) {
    for(k=1;k<=vpts;k++) {       /* all of the vpoints */
      for(d=1;d<=dims;d++)
	x[d]=0.0;
      for(d=1;d<=dims;d++)
	for(aa=1;aa<=ends;aa++) {
	  a=sidenodes[side][aa];
	  x[d] += E->X[lev][CPPR][d][E->IEN[lev][CPPR][el].node[a]]
	    *E->M.vpt[GMVINDEX(aa,k)];

	}

      rr = sqrt(x[1]*x[1]+x[2]*x[2]+x[3]*x[3]);
      tt = acos(x[3]/rr);
      ff = myatan(x[2],x[1]);
#ifdef ALLOW_ELLIPTICAL
      tt = theta_g(acos(x[3]/rr),E);
#else
      tt = acos(x[3]/rr);
#endif

      costt = cos(tt);		
      cosff = cos(ff);
      sintt = sin(tt);
      sinff = sin(ff);

      u[1][1] = costt*cosff; u[1][2] = costt*sinff;  u[1][3] =-sintt;
      u[2][1] =-sinff;       u[2][2] = cosff;        u[2][3] = 0.0;
      u[3][1] = sintt*cosff; u[3][2] = sintt*sinff;  u[3][3] = costt;

      ux[1][1][1] =-sintt*cosff;  ux[1][1][2] =-sintt*sinff;  ux[1][1][3] =-costt;
      ux[2][1][1] =-costt*sinff;  ux[2][1][2] = costt*cosff;  ux[2][1][3] =0.0;
      ux[1][2][1] =0.0;           ux[1][2][2] = 0.0;          ux[1][2][3] =0.0;
      ux[2][2][1] =-cosff;        ux[2][2][2] =-sinff;        ux[2][2][3] =0.0;
      ux[1][3][1] = costt*cosff;  ux[1][3][2] = costt*sinff;  ux[1][3][3] =-sintt;
      ux[2][3][1] =-sintt*sinff;  ux[2][3][2] = sintt*cosff;  ux[2][3][3] =0.0;

      for(aa=1;aa<=ends;aa++) {
	a=sidenodes[side][aa];

	lnode = E->IEN[lev][CPPR][el].node[a];
	sintt = E->SinCos[lev][CPPR][0][lnode];
	sinff = E->SinCos[lev][CPPR][1][lnode];
	costt = E->SinCos[lev][CPPR][2][lnode];
	cosff = E->SinCos[lev][CPPR][3][lnode];

	ua[1][1] = costt*cosff; ua[1][2] = costt*sinff;  ua[1][3] =-sintt;
	ua[2][1] =-sinff;       ua[2][2] = cosff;        ua[2][3] = 0.0;
	ua[3][1] = sintt*cosff; ua[3][2] = sintt*sinff;  ua[3][3] = costt;

	for (i=1;i<=dims;i++)
	  for (j=1;j<=dims;j++)   {
	    cc->vpt[BVINDEX(i,j,a,k)] =
	      ua[j][1]*u[i][1]+ua[j][2]*u[i][2]+ua[j][3]*u[i][3];
	    ccx->vpt[BVXINDEX(i,j,1,a,k)] =
	      ua[j][1]*ux[1][i][1]+ua[j][2]*ux[1][i][2]+ua[j][3]*ux[1][i][3];
	    ccx->vpt[BVXINDEX(i,j,2,a,k)] =
	      ua[j][1]*ux[2][i][1]+ua[j][2]*ux[2][i][2]+ua[j][3]*ux[2][i][3];
	  }

      }      /* end for local node */
    }        /* end for int points */
  }    /* end if */
  else {
    for(k=1;k<=ppts;k++) {       /* all of the ppoints */
      for(d=1;d<=E->mesh.nsd;d++)
       	x[d]=0.0;
      for(a=1;a<=ends;a++) {
       	aa=sidenodes[side][a];
       	x[d] += E->X[lev][CPPR][d][E->IEN[lev][CPPR][el].node[aa]]
       	  *E->M.ppt[GMPINDEX(a,k)];
      }
      rr = sqrt(x[1]*x[1]+x[2]*x[2]+x[3]*x[3]);
      ff = myatan(x[2],x[1]);

#ifdef ALLOW_ELLIPTICAL
      tt = theta_g(acos(x[3]/rr),E);
#else
      tt = acos(x[3]/rr);	
#endif
      costt = cos(tt);
      cosff = cos(ff);
      sintt = sin(tt);
      sinff = sin(ff);

      u[1][1] = costt*cosff; u[1][2] = costt*sinff;  u[1][3] =-sintt;
      u[2][1] =-sinff;       u[2][2] = cosff;        u[2][3] = 0.0;
      u[3][1] = sintt*cosff; u[3][2] = sintt*sinff;  u[3][3] = costt;

      ux[1][1][1] =-sintt*cosff;  ux[1][1][2] =-sintt*sinff;  ux[1][1][3] =-costt;
      ux[2][1][1] =-costt*sinff;  ux[2][1][2] = costt*cosff;  ux[2][1][3] =0.0;
      ux[1][2][1] =0.0;           ux[1][2][2] = 0.0;          ux[1][2][3] =0.0;
      ux[2][2][1] =-cosff;        ux[2][2][2] =-sinff;        ux[2][2][3] =0.0;
      ux[1][3][1] = costt*cosff;  ux[1][3][2] = costt*sinff;  ux[1][3][3] =-sintt;
      ux[2][3][1] =-sintt*sinff;  ux[2][3][2] = sintt*cosff;  ux[2][3][3] =0.0;

      for(a=1;a<=ends;a++)   {
	aa=sidenodes[side][a];

	lnode = E->IEN[lev][CPPR][el].node[aa];
	sintt = E->SinCos[lev][CPPR][0][lnode];
	sinff = E->SinCos[lev][CPPR][1][lnode];
	costt = E->SinCos[lev][CPPR][2][lnode];
	cosff = E->SinCos[lev][CPPR][3][lnode];

	ua[1][1] = costt*cosff; ua[1][2] = costt*sinff;  ua[1][3] =-sintt;
	ua[2][1] =-sinff;       ua[2][2] = cosff;        ua[2][3] = 0.0;
	ua[3][1] = sintt*cosff; ua[3][2] = sintt*sinff;  ua[3][3] = costt;

	for (i=1;i<=E->mesh.nsd;i++) {
	  for (j=1;j<=E->mesh.nsd;j++) {
	    cc->ppt[BPINDEX(i,j,a,k)] =
	      ua[j][1]*u[i][1]+ua[j][2]*u[i][2]+ua[j][3]*u[i][3];
	    ccx->ppt[BPXINDEX(i,j,1,a,k)] =
	      ua[j][1]*ux[1][i][1]+ua[j][2]*ux[1][i][2]+ua[j][3]*ux[1][i][3];
	    ccx->ppt[BPXINDEX(i,j,2,a,k)] =
	      ua[j][1]*ux[2][i][1]+ua[j][2]*ux[2][i][2]+ua[j][3]*ux[2][i][3];
	  }
	}
      }      /* end for local node */
    }      /* end for int points */
  }      /* end if pressure  */
}


/* ======================================= */
void construct_c3x3matrix(E)
     struct All_variables *E;
{
  int m,a,i,j,k,d,es,el,nel_surface,lev,lnode;
  double cofactor(),myatan();
  double x[4],u[4][4],ux[3][4][4],ua[4][4];
  double costt,cosff,sintt,sinff,rr,tt,ff;

  const int dims=E->mesh.nsd;
  const int ends=enodes[dims];
  const int vpts=vpoints[dims];
  const int ppts=ppoints[dims];

 for (lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++) {
    nel_surface = E->lmesh.NEL[lev]/E->lmesh.ELZ[lev];
    for (es=1;es<=nel_surface;es++)        {

      el = es*E->lmesh.ELZ[lev];

      for(k=1;k<=vpts;k++)           {       /* all of the vpoints */
        for(d=1;d<=dims;d++)
          x[d]=0.0;

        for(d=1;d<=dims;d++)
          for(a=1;a<=ends;a++)
            x[d] += E->X[lev][CPPR][d][E->IEN[lev][CPPR][el].node[a]]
                   *E->N.vpt[GNVINDEX(a,k)];

        rr = sqrt(x[1]*x[1]+x[2]*x[2]+x[3]*x[3]);
	ff = myatan(x[2],x[1]);
#ifdef ALLOW_ELLIPTICAL
	tt = theta_g(acos(x[3]/rr),E);
#else
	tt = acos(x[3]/rr);
#endif

        costt = cos(tt);	
        cosff = cos(ff);
        sintt = sin(tt);
        sinff = sin(ff);

        u[1][1] = costt*cosff; u[1][2] = costt*sinff;  u[1][3] =-sintt;
        u[2][1] =-sinff;       u[2][2] = cosff;        u[2][3] = 0.0;
        u[3][1] = sintt*cosff; u[3][2] = sintt*sinff;  u[3][3] = costt;

        ux[1][1][1] =-sintt*cosff;  ux[1][1][2] =-sintt*sinff;  ux[1][1][3] =-costt;
        ux[2][1][1] =-costt*sinff;  ux[2][1][2] = costt*cosff;  ux[2][1][3] =0.0;
        ux[1][2][1] =0.0;           ux[1][2][2] = 0.0;          ux[1][2][3] =0.0;
        ux[2][2][1] =-cosff;        ux[2][2][2] =-sinff;        ux[2][2][3] =0.0;
        ux[1][3][1] = costt*cosff;  ux[1][3][2] = costt*sinff;  ux[1][3][3] =-sintt;
        ux[2][3][1] =-sintt*sinff;  ux[2][3][2] = sintt*cosff;  ux[2][3][3] =0.0;

        for(a=1;a<=ends;a++)   {

	  lnode = E->IEN[lev][CPPR][el].node[a];
	  sintt = E->SinCos[lev][CPPR][0][lnode];
	  sinff = E->SinCos[lev][CPPR][1][lnode];
	  costt = E->SinCos[lev][CPPR][2][lnode];
	  cosff = E->SinCos[lev][CPPR][3][lnode];

          ua[1][1] = costt*cosff; ua[1][2] = costt*sinff;  ua[1][3] =-sintt;
          ua[2][1] =-sinff;       ua[2][2] = cosff;        ua[2][3] = 0.0;
          ua[3][1] = sintt*cosff; ua[3][2] = sintt*sinff;  ua[3][3] = costt;

          for (i=1;i<=dims;i++)
            for (j=1;j<=dims;j++)   {
              E->CC[lev][CPPR][es].vpt[BVINDEX(i,j,a,k)] =
                    ua[j][1]*u[i][1]+ua[j][2]*u[i][2]+ua[j][3]*u[i][3];
              E->CCX[lev][CPPR][es].vpt[BVXINDEX(i,j,1,a,k)] =
                    ua[j][1]*ux[1][i][1]+ua[j][2]*ux[1][i][2]+ua[j][3]*ux[1][i][3];
              E->CCX[lev][CPPR][es].vpt[BVXINDEX(i,j,2,a,k)] =
                    ua[j][1]*ux[2][i][1]+ua[j][2]*ux[2][i][2]+ua[j][3]*ux[2][i][3];
              }
          }      /* end for local node */

        }        /* end for int points */

      for(k=1;k<=ppts;k++)           {       /* all of the ppoints */
        for(d=1;d<=dims;d++)
          x[d]=0.0;

        for(d=1;d<=dims;d++)
          for(a=1;a<=ends;a++)
            x[d] += E->X[lev][CPPR][d][E->IEN[lev][CPPR][el].node[a]]
                   *E->N.ppt[GNPINDEX(a,k)];

        rr = sqrt(x[1]*x[1]+x[2]*x[2]+x[3]*x[3]);
	ff = myatan(x[2],x[1]);
#ifdef ALLOW_ELLIPTICAL
	tt = theta_g(acos(x[3]/rr),E);
#else
	tt = acos(x[3]/rr);
#endif

        costt = cos(tt);	
        cosff = cos(ff);
        sintt = sin(tt);
        sinff = sin(ff);

        u[1][1] = costt*cosff; u[1][2] = costt*sinff;  u[1][3] =-sintt;
        u[2][1] =-sinff;       u[2][2] = cosff;        u[2][3] = 0.0;
        u[3][1] = sintt*cosff; u[3][2] = sintt*sinff;  u[3][3] = costt;

        ux[1][1][1] =-sintt*cosff;  ux[1][1][2] =-sintt*sinff;  ux[1][1][3] =-costt;
        ux[2][1][1] =-costt*sinff;  ux[2][1][2] = costt*cosff;  ux[2][1][3] =0.0;
        ux[1][2][1] =0.0;           ux[1][2][2] = 0.0;          ux[1][2][3] =0.0;
        ux[2][2][1] =-cosff;        ux[2][2][2] =-sinff;        ux[2][2][3] =0.0;
        ux[1][3][1] = costt*cosff;  ux[1][3][2] = costt*sinff;  ux[1][3][3] =-sintt;
        ux[2][3][1] =-sintt*sinff;  ux[2][3][2] = sintt*cosff;  ux[2][3][3] =0.0;

        for(a=1;a<=ends;a++)   {

	  lnode = E->IEN[lev][CPPR][el].node[a];
	  sintt = E->SinCos[lev][CPPR][0][lnode];
	  sinff = E->SinCos[lev][CPPR][1][lnode];
	  costt = E->SinCos[lev][CPPR][2][lnode];
	  cosff = E->SinCos[lev][CPPR][3][lnode];


          ua[1][1] = costt*cosff; ua[1][2] = costt*sinff;  ua[1][3] =-sintt;
          ua[2][1] =-sinff;       ua[2][2] = cosff;        ua[2][3] = 0.0;
          ua[3][1] = sintt*cosff; ua[3][2] = sintt*sinff;  ua[3][3] = costt;

          for (i=1;i<=dims;i++)
            for (j=1;j<=dims;j++)   {
              E->CC[lev][CPPR][es].ppt[BPINDEX(i,j,a,k)] =
                    ua[j][1]*u[i][1]+ua[j][2]*u[i][2]+ua[j][3]*u[i][3];
              E->CCX[lev][CPPR][es].ppt[BPXINDEX(i,j,1,a,k)] =
                    ua[j][1]*ux[1][i][1]+ua[j][2]*ux[1][i][2]+ua[j][3]*ux[1][i][3];
              E->CCX[lev][CPPR][es].ppt[BPXINDEX(i,j,2,a,k)] =
                    ua[j][1]*ux[2][i][1]+ua[j][2]*ux[2][i][2]+ua[j][3]*ux[2][i][3];
              }

          }      /* end for local node */

        }        /* end for int points */


      }         /* end for es */
    }           /* end for lev */
}



/*  ==========================================
    construct the lumped mass matrix. The full
    matrix is the FE integration of the density
    field. The lumped version is the diagonal
    matrix obtained by letting the shape function
    Na be delta(a,b)
    ========================================== */

void mass_matrix(struct All_variables *E)
{
    int m,node,i,nint,e,lev;
    int n[9], nz;
    double myatan(),area,centre[4],temp[9],temp2[9],dx1,dx2,dx3;

    const int vpts=vpoints[E->mesh.nsd];

    /* ECO .size can also be defined here */

    for(lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)  {

            for(node=1;node<=E->lmesh.NNO[lev];node++)
                E->MASS[lev][node] = 0.0;

            for(e=1;e<=E->lmesh.NEL[lev];e++)  {

                area = centre[1] = centre[2] = centre[3] = 0.0;

                for(node=1;node<=enodes[E->mesh.nsd];node++)
                    n[node] = E->IEN[lev][CPPR][e].node[node];

                for(i=1;i<=E->mesh.nsd;i++)  {
                    for(node=1;node<=enodes[E->mesh.nsd];node++)
                        centre[i] += E->X[lev][CPPR][i][n[node]];

                    centre[i] = centre[i]/enodes[E->mesh.nsd];
                }     /* end for i */

                /* dx3=radius, dx1=theta, dx2=phi */
                dx3 = sqrt(centre[1]*centre[1]+centre[2]*centre[2]+centre[3]*centre[3]);
                dx1 = acos( centre[3]/dx3 );
                dx2 = myatan(centre[2],centre[1]);

                /* center of this element in the spherical coordinate */
                E->ECO[lev][CPPR][e].centre[1] = dx1;
                E->ECO[lev][CPPR][e].centre[2] = dx2;
                E->ECO[lev][CPPR][e].centre[3] = dx3;

                /* delta(theta) of this element */
                dx1 = max( fabs(E->SX[lev][CPPR][1][n[3]]-E->SX[lev][CPPR][1][n[1]]),
                           fabs(E->SX[lev][CPPR][1][n[2]]-E->SX[lev][CPPR][1][n[4]]) );

                /* length of this element in the theta-direction */
                E->ECO[lev][CPPR][e].size[1] = dx1*E->ECO[lev][CPPR][e].centre[3];

                /* delta(phi) of this element */
                dx1 = fabs(E->SX[lev][CPPR][2][n[3]]-E->SX[lev][CPPR][2][n[1]]);
                if (dx1>M_PI)
                    dx1 = min(E->SX[lev][CPPR][2][n[3]],E->SX[lev][CPPR][2][n[1]]) + 2.0*M_PI -
                        max(E->SX[lev][CPPR][2][n[3]],E->SX[lev][CPPR][2][n[1]]) ;

                dx2 = fabs(E->SX[lev][CPPR][2][n[2]]-E->SX[lev][CPPR][2][n[4]]);
                if (dx2>M_PI)
                    dx2 = min(E->SX[lev][CPPR][2][n[2]],E->SX[lev][CPPR][2][n[4]]) + 2.0*M_PI -
                        max(E->SX[lev][CPPR][2][n[2]],E->SX[lev][CPPR][2][n[4]]) ;

                dx2 = max(dx1,dx2);

                /* length of this element in the phi-direction */
                E->ECO[lev][CPPR][e].size[2] = dx2*E->ECO[lev][CPPR][e].centre[3]
                    *sin(E->ECO[lev][CPPR][e].centre[1]);

                /* delta(radius) of this element */
                dx3 = 0.25*(fabs(E->SX[lev][CPPR][3][n[5]]+E->SX[lev][CPPR][3][n[6]]
                                 +E->SX[lev][CPPR][3][n[7]]+E->SX[lev][CPPR][3][n[8]]
                                 -E->SX[lev][CPPR][3][n[1]]-E->SX[lev][CPPR][3][n[2]]
                                 -E->SX[lev][CPPR][3][n[3]]-E->SX[lev][CPPR][3][n[4]]));

                /* length of this element in the radius-direction */
                E->ECO[lev][CPPR][e].size[3] = dx3;

                /* volume (area in 2D) of this element */
                for(nint=1;nint<=vpts;nint++)
                    area += g_point[nint].weight[E->mesh.nsd-1] * E->GDA[lev][CPPR][e].vpt[nint];
                E->ECO[lev][CPPR][e].area = area;

                for(node=1;node<=enodes[E->mesh.nsd];node++)  {
                    temp[node] = 0.0;
                    for(nint=1;nint<=vpts;nint++)
                        temp[node] += E->GDA[lev][CPPR][e].vpt[nint]*g_point[nint].weight[E->mesh.nsd-1]
                            *E->N.vpt[GNVINDEX(node,nint)];       /* int Na dV */
                }

                for(node=1;node<=enodes[E->mesh.nsd];node++)
                    E->MASS[lev][E->IEN[lev][CPPR][e].node[node]] += temp[node];

                /* weight of each node, equivalent to pmass in ConMan */
                for(node=1;node<=enodes[E->mesh.nsd];node++)
                    E->TWW[lev][CPPR][e].node[node] = temp[node];


            } /* end of ele*/


        if(lev == E->mesh.levmax)
            for(node=1;node<=E->lmesh.NNO[lev];node++)
                E->NMass[node] = E->MASS[lev][node];

        if (E->control.NMULTIGRID||E->mesh.levmax==lev)
            (E->exchange_node_d)(E,E->MASS[lev],lev);

        for(node=1;node<=E->lmesh.NNO[lev];node++)
            E->MASS[lev][node] = 1.0/E->MASS[lev][node];

    } /* end of for lev */


        for(node=1;node<=E->lmesh.nno;node++)
            E->TMass[node] = 0.0;

        for(e=1;e<=E->lmesh.nel;e++)  {
            for(node=1;node<=enodes[E->mesh.nsd];node++) {
                temp[node] = 0.0;
                nz = ((E->ien[e].node[node]-1) % E->lmesh.noz) + 1;
                for(nint=1;nint<=vpts;nint++)
                    temp[node] += E->refstate.rho[nz]
                        * E->refstate.heat_capacity[nz]
                        * E->gDA[CPPR][e].vpt[nint]
                        * g_point[nint].weight[E->mesh.nsd-1]
                        * E->N.vpt[GNVINDEX(node,nint)];
            }

            /* lumped mass matrix, equivalent to tmass in ConMan */
            for(node=1;node<=enodes[E->mesh.nsd];node++)
                E->TMass[E->ien[e].node[node]] += temp[node];

        } /* end of for e */

    (E->exchange_node_d)(E,E->TMass,E->mesh.levmax);
      for(node=1;node<=E->lmesh.nno;node++)
          E->TMass[node] = 1.0 / E->TMass[node];


    /* compute volume of this processor mesh and the whole mesh */
    E->lmesh.volume = 0;
    E->mesh.volume = 0;

    for(e=1;e<=E->lmesh.nel;e++)
        E->lmesh.volume += E->eco[e].area;

    MPI_Allreduce(&E->lmesh.volume, &E->mesh.volume, 1, MPI_DOUBLE,
                  MPI_SUM, E->parallel.world);


    if (E->control.verbose)  {
        fprintf(E->fp_out, "rank=%d my_volume=%e total_volume=%e\n",
                E->parallel.me, E->lmesh.volume, E->mesh.volume);

        for(lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)  {
            fprintf(E->fp_out,"output_mass lev=%d\n",lev);
                fprintf(E->fp_out,"m=%d %d \n",E->sphere.capid[CPPR],m);
                for(e=1;e<=E->lmesh.NEL[lev];e++)
                    fprintf(E->fp_out,"%d %g \n",e,E->ECO[lev][CPPR][e].area);
                for (node=1;node<=E->lmesh.NNO[lev];node++)
                    fprintf(E->fp_out,"Mass[%d]= %g \n",node,E->MASS[lev][node]);
        }

            fprintf(E->fp_out,"m=%d %d \n",E->sphere.capid[CPPR],m);
            for (node=1;node<=E->lmesh.nno;node++)
                fprintf(E->fp_out,"TMass[%d]= %g \n",node,E->TMass[node]);
        fflush(E->fp_out);
    }
}
