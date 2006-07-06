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

void full_coord_of_cap(E,m,icap)
   struct All_variables *E;
   int icap,m;
  {

  int i,j,k,lev,temp,elx,ely,nox,noy,noz,node,nodes;
  int lelx,lely,lnox,lnoy,lnoz;
  double x[5],y[5],z[5],xx[5],yy[5],zz[5];
  double *theta1,*fi1,*theta2,*fi2,*theta,*fi,*SX[2];
  double myatan();

  void even_divide_arc12();

  temp = max(E->mesh.NOY[E->mesh.levmax],E->mesh.NOX[E->mesh.levmax]);

  theta1 = (double *)malloc((temp+1)*sizeof(double));
  fi1    = (double *)malloc((temp+1)*sizeof(double));
  theta2 = (double *)malloc((temp+1)*sizeof(double));
  fi2    = (double *)malloc((temp+1)*sizeof(double));
  theta  = (double *)malloc((temp+1)*sizeof(double));
  fi     = (double *)malloc((temp+1)*sizeof(double));

  temp = E->mesh.NOY[E->mesh.levmax]*E->mesh.NOX[E->mesh.levmax]; /* allocate enough for the entire cap */

  SX[0]  = (double *)malloc((temp+1)*sizeof(double));
  SX[1]  = (double *)malloc((temp+1)*sizeof(double));

  for (i=1;i<=4;i++)    {
     x[i] = sin(E->sphere.cap[icap].theta[i])*cos(E->sphere.cap[icap].fi[i]);
     y[i] = sin(E->sphere.cap[icap].theta[i])*sin(E->sphere.cap[icap].fi[i]);
     z[i] = cos(E->sphere.cap[icap].theta[i]);
     }
  
  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)  {

     elx = E->lmesh.ELX[lev]*E->parallel.nprocx;
     ely = E->lmesh.ELY[lev]*E->parallel.nprocy;
     nox = elx+1;
     noy = ely+1;

     lelx = E->lmesh.ELX[lev];
     lely = E->lmesh.ELY[lev];
     lnox = lelx+1;
     lnoy = lely+1;
     lnoz = E->lmesh.NOZ[lev];
        /* evenly divide arc linking 1 and 2, and the arc linking 4 and 3 */

     even_divide_arc12(elx,x[1],y[1],z[1],x[2],y[2],z[2],theta1,fi1);
     even_divide_arc12(elx,x[4],y[4],z[4],x[3],y[3],z[3],theta2,fi2);

     for (j=1;j<=nox;j++)   {

         /* pick up the two ends  */
        xx[1] = sin(theta1[j])*cos(fi1[j]);
        yy[1] = sin(theta1[j])*sin(fi1[j]);
        zz[1] = cos(theta1[j]);
        xx[2] = sin(theta2[j])*cos(fi2[j]);
        yy[2] = sin(theta2[j])*sin(fi2[j]);
        zz[2] = cos(theta2[j]);


        even_divide_arc12(ely,xx[1],yy[1],zz[1],xx[2],yy[2],zz[2],theta,fi);

        for (k=1;k<=noy;k++)   {
           nodes = j + (k-1)*nox;
           SX[0][nodes] = theta[k];
           SX[1][nodes] = fi[k];
           }


        }       /* end for j */

                /* get the coordinates for the entire cap  */

        for (j=1;j<=lnox;j++)
          for (k=1;k<=lnoy;k++)          {
             nodes = (j+(E->lmesh.NXS[lev]-1))+(k-1+(E->lmesh.NYS[lev]-1))*nox;
             for (i=1;i<=lnoz;i++)          {
                node = i + (j-1)*lnoz + (k-1)*lnox*lnoz;

                     /*   theta,fi,and r coordinates   */
                E->SX[lev][m][1][node] = SX[0][nodes];
                E->SX[lev][m][2][node] = SX[1][nodes];
                E->SX[lev][m][3][node] = E->sphere.R[lev][i];

                     /*   x,y,and z oordinates   */
                E->X[lev][m][1][node] = 
                            E->sphere.R[lev][i]*sin(SX[0][nodes])*cos(SX[1][nodes]);
                E->X[lev][m][2][node] = 
                            E->sphere.R[lev][i]*sin(SX[0][nodes])*sin(SX[1][nodes]);
                E->X[lev][m][3][node] = 
                            E->sphere.R[lev][i]*cos(SX[0][nodes]);
                }
             }

     }       /* end for lev */

  free ((void *)SX[0]);
  free ((void *)SX[1]);
  free ((void *)theta );
  free ((void *)theta1);
  free ((void *)theta2);
  free ((void *)fi    );
  free ((void *)fi1   );
  free ((void *)fi2   );

  return;
  }

