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

void regional_coord_of_cap(struct All_variables *E, int m, int icap)
  {

  int i,j,k,lev,temp,elx,ely,nox,noy,noz,node,nodes;
  int nprocxl,nprocyl,nproczl;
  int nnproc;
  int gnox,gnoy,gnoz;
  int nodesx,nodesy;
  char output_file[255];
  char a[100];
  int nn,step;
  FILE *fp;
  float *theta1[MAX_LEVELS],*fi1[MAX_LEVELS];
  double *SX[2];
  double *tt,*ff;
  double dt,df;
  double myatan();
  void parallel_process_termination();
  void myerror();

  void even_divide_arc12();

  m=1;

  gnox=E->mesh.nox;
  gnoy=E->mesh.noy;
  gnoz=E->mesh.noz;
  nox=E->lmesh.nox;
  noy=E->lmesh.noy;
  noz=E->lmesh.noz;

  nprocxl=E->parallel.nprocx;
  nprocyl=E->parallel.nprocy;
  nproczl=E->parallel.nprocz;
  nnproc=nprocyl*nprocxl*nproczl;
  temp = citmax(E->mesh.NOY[E->mesh.levmax],E->mesh.NOX[E->mesh.levmax]);

  /* define the cap corners */
  E->sphere.cap[1].theta[1] = E->control.theta_min;
  E->sphere.cap[1].theta[2] = E->control.theta_max;
  E->sphere.cap[1].theta[3] = E->control.theta_max;
  E->sphere.cap[1].theta[4] = E->control.theta_min;
  E->sphere.cap[1].fi[1] = E->control.fi_min;
  E->sphere.cap[1].fi[2] = E->control.fi_min;
  E->sphere.cap[1].fi[3] = E->control.fi_max;
  E->sphere.cap[1].fi[4] = E->control.fi_max;

  if(E->control.coor==1) {

    /* read in node locations from file */

    for(i=E->mesh.gridmin;i<=E->mesh.gridmax;i++)  {
      theta1[i] = (float *)malloc((temp+1)*sizeof(float));
      fi1[i]    = (float *)malloc((temp+1)*sizeof(float));
    }
    
    temp = E->mesh.NOY[E->mesh.levmax]*E->mesh.NOX[E->mesh.levmax];
    
    sprintf(output_file,"%s",E->control.coor_file);
    fp=fopen(output_file,"r");
    if (fp == NULL) {
      fprintf(E->fp,"(Sphere_related #1) Cannot open %s\n",output_file);
      exit(8);
    }

    fscanf(fp,"%s %d",a,&nn);
    for(i=1;i<=gnox;i++) {
        if(fscanf(fp,"%d %e",&nn,&theta1[E->mesh.gridmax][i]) != 2) {
            fprintf(E->fp,"Error while reading coor_file '%s'\n",output_file);
            exit(8);
        }
    }
    E->control.theta_min = theta1[E->mesh.gridmax][1];
    E->control.theta_max = theta1[E->mesh.gridmax][gnox];
    
    fscanf(fp,"%s %d",a,&nn);
    for(i=1;i<=gnoy;i++)  {
        if(fscanf(fp,"%d %e",&nn,&fi1[E->mesh.gridmax][i]) != 2) {
            fprintf(E->fp,"Error while reading coor_file '%s'\n",output_file);
            exit(8);
        }
    }
    E->control.fi_min = fi1[E->mesh.gridmax][1];
    E->control.fi_max = fi1[E->mesh.gridmax][gnoy];
    
    fclose(fp);
    
    /* redefine the cap corners */
    E->sphere.cap[1].theta[1] = E->control.theta_min;
    E->sphere.cap[1].theta[2] = E->control.theta_max;
    E->sphere.cap[1].theta[3] = E->control.theta_max;
    E->sphere.cap[1].theta[4] = E->control.theta_min;
    E->sphere.cap[1].fi[1] = E->control.fi_min;
    E->sphere.cap[1].fi[2] = E->control.fi_min;
    E->sphere.cap[1].fi[3] = E->control.fi_max;
    E->sphere.cap[1].fi[4] = E->control.fi_max;
    
    for (lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++)  {
      
      if (E->control.NMULTIGRID)
        step = (int) pow(2.0,(double)(E->mesh.levmax-lev));
      else
        step = 1;
      
      for (i=1;i<=E->mesh.NOX[lev];i++)
	theta1[lev][i] = theta1[E->mesh.gridmax][(i-1)*step+1];
      
      for (i=1;i<=E->mesh.NOY[lev];i++)
	fi1[lev][i] = fi1[E->mesh.gridmax][(i-1)*step+1];
      
    }


  for (lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++)  {
     elx = E->lmesh.ELX[lev];
     ely = E->lmesh.ELY[lev];
     nox = E->lmesh.NOX[lev];
     noy = E->lmesh.NOY[lev];
     noz = E->lmesh.NOZ[lev];
        /* evenly divide arc linking 1 and 2, and the arc linking 4 and 3 */

                /* get the coordinates for the entire cap  */

        for (j=1;j<=nox;j++)
          for (k=1;k<=noy;k++)          {
             nodesx = E->lmesh.NXS[lev]+j-1;
             nodesy = E->lmesh.NYS[lev]+k-1;

             for (i=1;i<=noz;i++)          {
                node = i + (j-1)*noz + (k-1)*nox*noz;

                     /*   theta,fi,and r coordinates   */
                E->SX[lev][m][1][node] = theta1[lev][nodesx];
                E->SX[lev][m][2][node] = fi1[lev][nodesy];
                E->SX[lev][m][3][node] = E->sphere.R[lev][i];

                     /*   x,y,and z oordinates   */
                E->X[lev][m][1][node] =
                            E->sphere.R[lev][i]*sin(theta1[lev][nodesx])*cos(fi1[lev][nodesy]);
                E->X[lev][m][2][node] =
                            E->sphere.R[lev][i]*sin(theta1[lev][nodesx])*sin(fi1[lev][nodesy]);
                E->X[lev][m][3][node] =
                            E->sphere.R[lev][i]*cos(theta1[lev][nodesx]);
                }
             }

     }       /* end for lev */



  for (lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++)  {
  free ((void *)theta1[lev]);
  free ((void *)fi1[lev]   );
   }

  } /* end of coord = 1 */
  
 else if((E->control.coor==0) || (E->control.coor==2)|| (E->control.coor==3))   {

  /*
  for(i=1;i<=5;i++)  {
  x[i] = (double *) malloc((E->parallel.nproc+1)*sizeof(double));
  y[i] = (double *) malloc((E->parallel.nproc+1)*sizeof(double));
  z[i] = (double *) malloc((E->parallel.nproc+1)*sizeof(double));
  */
  tt = (double *) malloc((4+1)*sizeof(double));
  ff = (double *) malloc((4+1)*sizeof(double));


  temp = E->lmesh.NOY[E->mesh.levmax]*E->lmesh.NOX[E->mesh.levmax];

  SX[0]  = (double *)malloc((temp+1)*sizeof(double));
  SX[1]  = (double *)malloc((temp+1)*sizeof(double));


     tt[1] = E->sphere.cap[m].theta[1]+(E->sphere.cap[m].theta[2] -E->sphere.cap[m].theta[1])/nprocxl*(E->parallel.me_loc[1]);
     tt[2] = E->sphere.cap[m].theta[1]+(E->sphere.cap[m].theta[2] -E->sphere.cap[m].theta[1])/nprocxl*(E->parallel.me_loc[1]+1);
     tt[3] = tt[2];
     tt[4] = tt[1];
     ff[1] = E->sphere.cap[m].fi[1]+(E->sphere.cap[m].fi[4] -E->sphere.cap[1].fi[1])/nprocyl*(E->parallel.me_loc[2]);
     ff[2] = ff[1];
     ff[3] = E->sphere.cap[m].fi[1]+(E->sphere.cap[m].fi[4] -E->sphere.cap[1].fi[1])/nprocyl*(E->parallel.me_loc[2]+1);
     ff[4] = ff[3];


  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)  {

     elx = E->lmesh.ELX[lev];
     ely = E->lmesh.ELY[lev];
     nox = E->lmesh.NOX[lev];
     noy = E->lmesh.NOY[lev];
     noz = E->lmesh.NOZ[lev];
        /* evenly divide arc linking 1 and 2, and the arc linking 4 and 3 */

      for(j=1;j<=nox;j++) {
       dt=(tt[3]-tt[1])/elx;
       df=(ff[3]-ff[1])/ely;

        for (k=1;k<=noy;k++)   {
           nodes = j + (k-1)*nox;
           SX[0][nodes] = tt[1]+dt*(j-1);
           SX[1][nodes] = ff[1]+df*(k-1);
           }

        }       /* end for j */

                /* get the coordinates for the entire cap  */

        for (j=1;j<=nox;j++)
          for (k=1;k<=noy;k++)          {
             nodes = j + (k-1)*nox;
             for (i=1;i<=noz;i++)          {
                node = i + (j-1)*noz + (k-1)*nox*noz;

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
  free ((void *)tt);
  free ((void *)ff);
}

  return;
  }

