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


/* Functions relating to the building and use of mesh locations ... */

#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"


/**************************************************************/
/* This function transforms theta and phi to new coords       */
/* u and v using gnomonic projection.                         */
/* See http://mathworld.wolfram.com/GnomonicProjection.html   */

void spherical_to_uv2(double center[2], int len,
                      double *theta, double *phi,
                      double *u, double *v)
{
    double theta_f, phi_f;
    double cos_tf, sin_tf;
    double cosc, cost, sint, cosp2, sinp2;
    int i;

    /* theta_f and phi_f are the reference points of the cap */

    theta_f = center[0];
    phi_f = center[1];

    cos_tf = cos(theta_f);
    sin_tf = sin(theta_f);

    for(i=0; i<len; i++) {
        cost = cos(theta[i]);
        sint = sin(theta[i]);

        cosp2 = cos(phi[i] - phi_f);
        sinp2 = sin(phi[i] - phi_f);

        cosc = cos_tf * cost + sin_tf * sint * cosp2;
        cosc = 1.0 / cosc;

        u[i] = sint * sinp2 * cosc;
        v[i] = (sin_tf * cost - cos_tf * sint * cosp2) * cosc;
    }
    return;
}


/**************************************************************/
/* This function transforms u and v to spherical coord        */
/* theta and phi using inverse gnomonic projection.           */
/* See http://mathworld.wolfram.com/GnomonicProjection.html   */

void uv_to_spherical(double center[2], int len,
                     double *u, double *v,
                     double *theta, double *phi)
{
    double theta_f, phi_f, cos_tf, sin_tf;
    double x, y, r, c;
    double cosc, sinc;
    int i;

    /* theta_f and phi_f are the reference points at the midpoint of the cap */

    theta_f = center[0];
    phi_f = center[1];

    cos_tf = cos(theta_f);
    sin_tf = sin(theta_f);

    for(i=0; i<len; i++) {
        x = u[i];
        y = v[i];
        r = sqrt(x*x + y*y);

        /* special case: r=0, then (u,v) is the reference point */
        if(r == 0) {
            theta[i] = theta_f;
            phi[i] = phi_f;
            continue;
        }

        /* c = atan(r); cosc = cos(c); sinc = sin(c);*/
        cosc = 1.0 / sqrt(1 + r*r);
        sinc = sqrt(1 - cosc*cosc);

        theta[i] = acos(cosc * cos_tf +
                        y * sinc * sin_tf / r);
        phi[i] = phi_f + atan(x * sinc /
                              (r * sin_tf * cosc - y * cos_tf * sinc));
    }
    return;
}


/* Find the intersection point of two lines    */
/* The lines are: (x[0], y[0]) to (x[1], y[1]) */
/*                (x[2], y[2]) to (x[3], y[3]) */
/* If found, the intersection point is stored  */
/*           in (px, py) and return 1          */
/* If not found, return 0                      */

static int find_intersection(double *x, double *y,
                             double *px, double *py)
{
    double a1, b1, c1;
    double a2, b2, c2;
    double denom;

    a1 = y[1] - y[0];
    b1 = x[0] - x[1];
    c1 = x[1]*y[0] - x[0]*y[1];

    a2 = y[3] - y[2];
    b2 = x[2] - x[3];
    c2 = x[3]*y[2] - x[2]*y[3];

    denom = a1*b2 - a2*b1;
    if (denom == 0) return 0; /* the lines are parallel! */

    *px = (b1*c2 - b2*c1)/denom;
    *py = (a2*c1 - a1*c2)/denom;
    return 1;
}


void full_coord_of_cap(struct All_variables *E, int m, int icap)
{
  int i, j, k, lev, temp, elx, ely;
  int node, snode, ns, step;
  int lelx, lely, lnox, lnoy;
  int lvnox, lvnoy, lvnoz;
  int ok;
  double x[5], y[5], z[5], center[3], referencep[2];
  double xx[5], yy[5];
  double *theta0, *fi0;
  double *tt1,  *tt2, *tt3, *tt4, *ff1, *ff2, *ff3, *ff4;
  double *u1, *u2, *u3, *u4, *v1, *v2, *v3, *v4;
  double *px, *py, *qx, *qy;
  double theta, fi, cost, sint, cosf, sinf, efac2,rfac;
  double a, b;
  double offset;

  temp = max(E->mesh.noy, E->mesh.nox);

  theta0 = (double *)malloc((temp+1)*sizeof(double));
  fi0    = (double *)malloc((temp+1)*sizeof(double));

  tt1    = (double *)malloc((temp+1)*sizeof(double));
  tt2    = (double *)malloc((temp+1)*sizeof(double));
  tt3    = (double *)malloc((temp+1)*sizeof(double));
  tt4    = (double *)malloc((temp+1)*sizeof(double));

  ff1    = (double *)malloc((temp+1)*sizeof(double));
  ff2    = (double *)malloc((temp+1)*sizeof(double));
  ff3    = (double *)malloc((temp+1)*sizeof(double));
  ff4    = (double *)malloc((temp+1)*sizeof(double));

  u1     = (double *)malloc((temp+1)*sizeof(double));
  u2     = (double *)malloc((temp+1)*sizeof(double));
  u3     = (double *)malloc((temp+1)*sizeof(double));
  u4     = (double *)malloc((temp+1)*sizeof(double));

  v1     = (double *)malloc((temp+1)*sizeof(double));
  v2     = (double *)malloc((temp+1)*sizeof(double));
  v3     = (double *)malloc((temp+1)*sizeof(double));
  v4     = (double *)malloc((temp+1)*sizeof(double));

  temp = E->mesh.noy * E->mesh.nox;
  px = (double *)malloc((temp+1)*sizeof(double));
  py = (double *)malloc((temp+1)*sizeof(double));
  qx = (double *)malloc((temp+1)*sizeof(double));
  qy = (double *)malloc((temp+1)*sizeof(double));

  /* define the cap corners */

  /* adjust the corner coordinates so that the size (surface area) of
     each cap is about the same. */
  offset = 9.736/180.0*M_PI;

  for (i=1;i<=4;i++)  {
    E->sphere.cap[(i-1)*3+1].theta[1] = 0.0;
    E->sphere.cap[(i-1)*3+1].theta[2] = M_PI/4.0+offset;
    E->sphere.cap[(i-1)*3+1].theta[3] = M_PI/2.0;
    E->sphere.cap[(i-1)*3+1].theta[4] = M_PI/4.0+offset;
    E->sphere.cap[(i-1)*3+1].fi[1] = 0.0;
    E->sphere.cap[(i-1)*3+1].fi[2] = (i-1)*M_PI/2.0;
    E->sphere.cap[(i-1)*3+1].fi[3] = (i-1)*M_PI/2.0 + M_PI/4.0;
    E->sphere.cap[(i-1)*3+1].fi[4] = i*M_PI/2.0;

    E->sphere.cap[(i-1)*3+2].theta[1] = M_PI/4.0+offset;
    E->sphere.cap[(i-1)*3+2].theta[2] = M_PI/2.0;
    E->sphere.cap[(i-1)*3+2].theta[3] = 3*M_PI/4.0-offset;
    E->sphere.cap[(i-1)*3+2].theta[4] = M_PI/2.0;
    E->sphere.cap[(i-1)*3+2].fi[1] = i*M_PI/2.0;
    E->sphere.cap[(i-1)*3+2].fi[2] = i*M_PI/2.0 - M_PI/4.0;
    E->sphere.cap[(i-1)*3+2].fi[3] = i*M_PI/2.0;
    E->sphere.cap[(i-1)*3+2].fi[4] = i*M_PI/2.0 + M_PI/4.0;
    }

  for (i=1;i<=4;i++)  {
    j = (i-1)*3;
    if (i==1) j=12;
    E->sphere.cap[j].theta[1] = M_PI/2.0;
    E->sphere.cap[j].theta[2] = 3*M_PI/4.0-offset;
    E->sphere.cap[j].theta[3] = M_PI;
    E->sphere.cap[j].theta[4] = 3*M_PI/4.0-offset;
    E->sphere.cap[j].fi[1] = (i-1)*M_PI/2.0 + M_PI/4.0;
    E->sphere.cap[j].fi[2] = (i-1)*M_PI/2.0;
    E->sphere.cap[j].fi[3] = 0.0;
    E->sphere.cap[j].fi[4] = i*M_PI/2.0;
    }


  /* 4 corners of the cap in Cartesian coordinates */
  /* the order of corners is: */
  /*  1 - 4 */
  /*  |   | */
  /*  2 - 3 */
  center[0] = center[1] = center[2] = 0;
#ifdef ALLOW_ELLIPTICAL
  for (i=1;i<=4;i++)    {	/* works for both elliptical and spherical */
    
    x[i] = E->data.ra * sin(E->sphere.cap[icap].theta[i])*cos(E->sphere.cap[icap].fi[i]);
    y[i] = E->data.ra * sin(E->sphere.cap[icap].theta[i])*sin(E->sphere.cap[icap].fi[i]);
    z[i] = E->data.rc * cos(E->sphere.cap[icap].theta[i]);
    
    center[0] += x[i];
    center[1] += y[i];
    center[2] += z[i];
  }
#else
  /* only spherical */
  for (i=1;i<=4;i++)    {
    x[i] = sin(E->sphere.cap[icap].theta[i])*cos(E->sphere.cap[icap].fi[i]);
    y[i] = sin(E->sphere.cap[icap].theta[i])*sin(E->sphere.cap[icap].fi[i]);
    z[i] = cos(E->sphere.cap[icap].theta[i]);
    center[0] += x[i];
    center[1] += y[i];
    center[2] += z[i];
  }
#endif

  center[0] *= 0.25;
  center[1] *= 0.25;
  center[2] *= 0.25;

  /* use the center as the reference point for gnomonic projection */
  referencep[0] = acos(center[2] /
                       sqrt(center[0]*center[0] +
                            center[1]*center[1] +
                            center[2]*center[2]));;
  referencep[1] = myatan(center[1], center[0]);


  lev = E->mesh.levmax;

     /* # of elements/nodes per cap */
     elx = E->lmesh.ELX[lev]*E->parallel.nprocx;
     ely = E->lmesh.ELY[lev]*E->parallel.nprocy;

     /* # of elements/nodes per processor */
     lelx = E->lmesh.ELX[lev];
     lely = E->lmesh.ELY[lev];
     lnox = lelx+1;
     lnoy = lely+1;

     /* evenly divide arc linking corner 1 and 2 */
     even_divide_arc12(elx,x[1],y[1],z[1],x[2],y[2],z[2],theta0,fi0);

     /* pick up only points within this processor */
     for (j=0, i=E->lmesh.nxs; j<lnox; j++, i++) {
         tt1[j] = theta0[i];
         ff1[j] = fi0[i];
     }

     /* evenly divide arc linking corner 4 and 3 */
     even_divide_arc12(elx,x[4],y[4],z[4],x[3],y[3],z[3],theta0,fi0);

     /* pick up only points within this processor */
     for (j=0, i=E->lmesh.nxs; j<lnox; j++, i++) {
         tt2[j] = theta0[i];
         ff2[j] = fi0[i];
     }

     /* evenly divide arc linking corner 1 and 4 */
     even_divide_arc12(ely,x[1],y[1],z[1],x[4],y[4],z[4],theta0,fi0);

     /* pick up only points within this processor */
     for (k=0, i=E->lmesh.nys; k<lnoy; k++, i++) {
         tt3[k] = theta0[i];
         ff3[k] = fi0[i];
     }

     /* evenly divide arc linking corner 2 and 3 */
     even_divide_arc12(ely,x[2],y[2],z[2],x[3],y[3],z[3],theta0,fi0);

     /* pick up only points within this processor */
     for (k=0, i=E->lmesh.nys; k<lnoy; k++, i++) {
         tt4[k] = theta0[i];
         ff4[k] = fi0[i];
     }

     /* compute the intersection point of these great circles */
     /* the point is first found in u-v space and project back */
     /* to theta-phi space later */

     spherical_to_uv2(referencep, lnox, tt1, ff1, u1, v1);
     spherical_to_uv2(referencep, lnox, tt2, ff2, u2, v2);
     spherical_to_uv2(referencep, lnoy, tt3, ff3, u3, v3);
     spherical_to_uv2(referencep, lnoy, tt4, ff4, u4, v4);

     snode = 0;
     for(k=0; k<lnoy; k++) {
         xx[2] = u3[k];
         yy[2] = v3[k];

         xx[3] = u4[k];
         yy[3] = v4[k];

         for(j=0; j<lnox; j++) {
             xx[0] = u1[j];
             yy[0] = v1[j];

             xx[1] = u2[j];
             yy[1] = v2[j];

             ok = find_intersection(xx, yy, &a, &b);
             if(!ok) {
                 fprintf(stderr, "Error(Full_coord_of_cap): cannot find intersection point! rank=%d, nx=%d, ny=%d\n", E->parallel.me, j, k);
		 fprintf(stderr, "L1: (%g, %g)-(%g, %g)  L2 (%g, %g)-(%g, %g)\n",
                         xx[0],yy[0],xx[1],yy[1],xx[2],yy[2],xx[3],yy[3]);
                 exit(10);
             }

             px[snode] = a;
             py[snode] = b;
             snode++;
         }
     }

     uv_to_spherical(referencep, snode, px, py, qx, qy);

     /* replace (qx, qy) by (tt?, ff?) for points on the edge */
     if(E->parallel.me_loc[2] == 0) {
         /* x boundary */
         for(k=0; k<lnox; k++) {
             i = k;
             qx[i] = tt1[k];
             qy[i] = ff1[k];
         }
     }

     if(E->parallel.me_loc[2] == E->parallel.nprocy-1) {
         /* x boundary */
         for(k=0; k<lnox; k++) {
             i = k + (lnoy - 1) * lnox;
             qx[i] = tt2[k];
             qy[i] = ff2[k];
         }
     }

     if(E->parallel.me_loc[1] == 0) {
         /* y boundary */
         for(k=0; k<lnoy; k++) {
             i = k * lnox;
             qx[i] = tt3[k];
             qy[i] = ff3[k];
         }
     }

     if(E->parallel.me_loc[1] == E->parallel.nprocx-1) {
         /* y boundary */
         for(k=0; k<lnoy; k++) {
             i = (k + 1) * lnox - 1;
             qx[i] = tt4[k];
             qy[i] = ff4[k];
         }
     }

#ifdef ALLOW_ELLIPTICAL     
     /* both spherical and elliptical */
     efac2 = E->data.ellipticity*(2.0 - E->data.ellipticity)/
       ((1.- E->data.ellipticity)*(1.-E->data.ellipticity));
     
     for (lev=E->mesh.levmax, step=1; lev>=E->mesh.levmin; lev--, step*=2) {
       /* store the node location in spherical and cartesian coordinates */
       
       lvnox = E->lmesh.NOX[lev];
       lvnoy = E->lmesh.NOY[lev];
       lvnoz = E->lmesh.NOZ[lev];
       
       node = 1;
       for (k=0; k<lvnoy; k++) {
	 for (j=0, ns=step*lnoy*k; j<lvnox; j++, ns+=step) {
	   theta = qx[ns];
	   fi = qy[ns];
	   
	   cost = cos(theta);
	   
	   rfac = E->data.ra*1./sqrt(1.0+efac2*cost*cost);
	   sint = sin(theta);
	   cosf = cos(fi);
	   sinf = sin(fi);
	   
	   for (i=1; i<=lvnoz; i++) {
	     /*   theta,fi,and r coordinates   */
	     E->SX[lev][m][1][node] = theta;
	     E->SX[lev][m][2][node] = fi;
	     E->SX[lev][m][3][node] = rfac * E->sphere.R[lev][i];
	     
	     /*   x,y,and z oordinates   */
	     E->X[lev][m][1][node] = E->data.ra * E->sphere.R[lev][i]*sint*cosf;
	     E->X[lev][m][2][node] = E->data.ra * E->sphere.R[lev][i]*sint*sinf;
	     E->X[lev][m][3][node] = E->data.rc * E->sphere.R[lev][i]*cost;
	     
	     node++;
	   }
	 }
       }
     } /* end for lev */
#else
     /* spherical */
     for (lev=E->mesh.levmax, step=1; lev>=E->mesh.levmin; lev--, step*=2) {
       /* store the node location in spherical and cartesian coordinates */
       
       lvnox = E->lmesh.NOX[lev];
       lvnoy = E->lmesh.NOY[lev];
       lvnoz = E->lmesh.NOZ[lev];

       node = 1;
       for (k=0; k<lvnoy; k++) {
	 for (j=0, ns=step*lnoy*k; j<lvnox; j++, ns+=step) {
	   theta = qx[ns];
	   fi =    qy[ns];
	   
	   cost = cos(theta);
	   sint = sin(theta);
	   cosf = cos(fi);
	   sinf = sin(fi);
	   
	   for (i=1; i<=lvnoz; i++) {
	     /*   theta,fi,and r coordinates   */
	     E->SX[lev][m][1][node] = theta;
	     E->SX[lev][m][2][node] = fi;
	     E->SX[lev][m][3][node] = E->sphere.R[lev][i];
	     
	     /*   x,y,and z oordinates   */
	     E->X[lev][m][1][node]  = E->sphere.R[lev][i]*sint*cosf;
	     E->X[lev][m][2][node]  = E->sphere.R[lev][i]*sint*sinf;
	     E->X[lev][m][3][node]  = E->sphere.R[lev][i]*cost;
	     
	     node++;
	   }
	 }
       }
     } /* end for lev */
#endif

  free ((void *)theta0);
  free ((void *)fi0   );

  free ((void *)tt1   );
  free ((void *)tt2   );
  free ((void *)tt3   );
  free ((void *)tt4   );

  free ((void *)ff1   );
  free ((void *)ff2   );
  free ((void *)ff3   );
  free ((void *)ff4   );

  free ((void *)u1    );
  free ((void *)u2    );
  free ((void *)u3    );
  free ((void *)u4    );

  free ((void *)v1    );
  free ((void *)v2    );
  free ((void *)v3    );
  free ((void *)v4    );

  free ((void *)px    );
  free ((void *)py    );
  free ((void *)qx    );
  free ((void *)qy    );


  return;
}

