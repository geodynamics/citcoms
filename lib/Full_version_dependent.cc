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

#include "full_version_dependent.h"

#include <math.h>

#include "global_defs.h"

#include "full_sphere_related.h"
#include "pan_problem_misc_functions.h"
#include "parallel_util.h"
#include "sphere_util.h"


/* =================================================
  rotate the mesh by a rotation matrix
 =================================================*/
static void full_rotate_mesh(struct All_variables *E, double dircos[4][4],
                             int m, int icap)
{
    int i,lev;
    double t[4];

    for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++) {
        for (i=1;i<=E->lmesh.NNO[lev];i++) {
            t[0] = E->X[lev][m][1][i]*dircos[1][1]+
                E->X[lev][m][2][i]*dircos[1][2]+
                E->X[lev][m][3][i]*dircos[1][3];
            t[1] = E->X[lev][m][1][i]*dircos[2][1]+
                E->X[lev][m][2][i]*dircos[2][2]+
                E->X[lev][m][3][i]*dircos[2][3];
            t[2] = E->X[lev][m][1][i]*dircos[3][1]+
                E->X[lev][m][2][i]*dircos[3][2]+
                E->X[lev][m][3][i]*dircos[3][3];

            E->X[lev][m][1][i] = t[0];
            E->X[lev][m][2][i] = t[1];
            E->X[lev][m][3][i] = t[2];
            E->SX[lev][m][1][i] = acos(t[2]/E->SX[lev][m][3][i]);
            E->SX[lev][m][2][i] = myatan(t[1],t[0]);
        }
    }    /* lev */

    return;
}

/* =================================================
   Standard node positions including mesh refinement

   =================================================  */

void full_node_locations(struct All_variables *E)
{
  int i,j,k,ii,lev;
  double ro,dr,*rr,*RR,fo,tg;
  double dircos[4][4];
  float tt1;
  int step,nn;
  char output_file[255], a[255];
  FILE *fp1;

  rr = (double *)  malloc((E->mesh.noz+1)*sizeof(double));
  RR = (double *)  malloc((E->mesh.noz+1)*sizeof(double));


  switch(E->control.coor){
  case 0:
    /* generate uniform mesh in radial direction */
    dr = (E->sphere.ro-E->sphere.ri)/(E->mesh.noz-1);

    for (k=1;k <= E->mesh.noz;k++)  {
      rr[k] = E->sphere.ri + (k-1)*dr;
    }
    break;
  case 1:			/* read nodal radii from file */
    sprintf(output_file,"%s",E->control.coor_file);
    fp1=fopen(output_file,"r");
    if (fp1 == NULL) {
      fprintf(E->fp,"(Nodal_mesh.c #1) Cannot open %s\n",output_file);
      exit(8);
    }
    fscanf(fp1,"%s %d",a,&i);
    for (k=1;k<=E->mesh.noz;k++)  {
      fscanf(fp1,"%d %f",&nn,&tt1);
      rr[k]=tt1;
    }

    fclose(fp1);
    break;
  case 2:
    /* higher radial spacing in top and bottom fractions */
    get_r_spacing_fine(rr,E);
    break;
  case 3:
    /* assign radial spacing CitcomCU style */
    get_r_spacing_at_levels(rr,E);
    break;
  default:
    myerror(E,"coor flag undefined in Full_version_dependent");
    break;
  }

  for (i=1;i<=E->lmesh.noz;i++)  {
    k = E->lmesh.nzs+i-1;
    RR[i] = rr[k];


  }


  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++) {

    if (E->control.NMULTIGRID||E->control.EMULTIGRID)
        step = (int) pow(2.0,(double)(E->mesh.levmax-lev));
    else
        step = 1;

      for (i=1;i<=E->lmesh.NOZ[lev];i++)
         E->sphere.R[lev][i] = RR[(i-1)*step+1];

    }          /* lev   */

  free ((void *) rr);
  free ((void *) RR);

  for (j=1;j<=E->sphere.caps_per_proc;j++)   {
     ii = E->sphere.capid[j];
     full_coord_of_cap(E,j,ii);
     }

  if (E->control.verbose) {
      for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)   {
          fprintf(E->fp_out,"output_coordinates before rotation %d \n",lev);
          for (j=1;j<=E->sphere.caps_per_proc;j++)
              for (i=1;i<=E->lmesh.NNO[lev];i++)
                  if(i%E->lmesh.NOZ[lev]==1)
                      fprintf(E->fp_out,"%d %d %g %g %g\n",j,i,E->SX[lev][j][1][i],E->SX[lev][j][2][i],E->SX[lev][j][3][i]);
      }
      fflush(E->fp_out);
  }

  /* rotate the mesh to avoid two poles on mesh points */

  ro = -0.5*(M_PI/4.0)/E->mesh.elx;
  fo = 0.0;

  dircos[1][1] = cos(ro)*cos(fo);
  dircos[1][2] = cos(ro)*sin(fo);
  dircos[1][3] = -sin(ro);
  dircos[2][1] = -sin(fo);
  dircos[2][2] = cos(fo);
  dircos[2][3] = 0.0;
  dircos[3][1] = sin(ro)*cos(fo);
  dircos[3][2] = sin(ro)*sin(fo);
  dircos[3][3] = cos(ro);

  for (j=1;j<=E->sphere.caps_per_proc;j++)   {
     ii = E->sphere.capid[j];
     full_rotate_mesh(E,dircos,j,ii);
     }

  if (E->control.verbose) {
      for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)   {
          fprintf(E->fp_out,"output_coordinates after rotation %d \n",lev);
          for (j=1;j<=E->sphere.caps_per_proc;j++)
              for (i=1;i<=E->lmesh.NNO[lev];i++)
                  if(i%E->lmesh.NOZ[lev]==1)
                      fprintf(E->fp_out,"%d %d %g %g %g\n",j,i,E->SX[lev][j][1][i],E->SX[lev][j][2][i],E->SX[lev][j][3][i]);
      }
      fflush(E->fp_out);
  }

  compute_angle_surf_area (E);   /* used for interpolation */
#ifdef ALLOW_ELLIPTICAL
  /* spherical or elliptical, correct theta to theta_g for local surface-normal theta  */
  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)
    for (j=1;j<=E->sphere.caps_per_proc;j++)
      for (i=1;i<=E->lmesh.NNO[lev];i++)  {
	tg = theta_g(E->SX[lev][j][1][i],E);
	E->SinCos[lev][j][0][i] = sin(tg); /*  */
	E->SinCos[lev][j][1][i] = sin(E->SX[lev][j][2][i]);
	E->SinCos[lev][j][2][i] = cos(tg);
	E->SinCos[lev][j][3][i] = cos(E->SX[lev][j][2][i]);
      }
#else
  /* spherical */
  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)
    for (j=1;j<=E->sphere.caps_per_proc;j++)
      for (i=1;i<=E->lmesh.NNO[lev];i++)  {
	E->SinCos[lev][j][0][i] = sin(E->SX[lev][j][1][i]); /* sin(theta) */
	E->SinCos[lev][j][1][i] = sin(E->SX[lev][j][2][i]); /* sin(phi) */
	E->SinCos[lev][j][2][i] = cos(E->SX[lev][j][1][i]); /* cos(theta) */
	E->SinCos[lev][j][3][i] = cos(E->SX[lev][j][2][i]); /* cos(phi) */
      }

#endif

  return;
}



/* setup boundary node and element arrays for bookkeeping */

void full_construct_boundary( struct All_variables *E)
{

  const int dims=E->mesh.nsd;

  int m, i, j, k, d, el, count;

  /* boundary = top + bottom */
  int max_size = 2*E->lmesh.elx*E->lmesh.ely + 1;
  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    E->boundary.element[m] = (int *)malloc(max_size*sizeof(int));

    for(d=1; d<=dims; d++)
      E->boundary.normal[m][d] = (int *)malloc(max_size*sizeof(int));
  }

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    count = 1;
    for(k=1; k<=E->lmesh.ely; k++)
      for(j=1; j<=E->lmesh.elx; j++) {
	if(E->parallel.me_loc[3] == 0) {
	  i = 1;
	  el = i + (j-1)*E->lmesh.elz + (k-1)*E->lmesh.elz*E->lmesh.elx;
	  E->boundary.element[m][count] = el;
	  E->boundary.normal[m][dims][count] = -1;
	  for(d=1; d<dims; d++)
	      E->boundary.normal[m][d][count] = 0;
	  ++count;
	}

	if(E->parallel.me_loc[3] == E->parallel.nprocz - 1) {
	  i = E->lmesh.elz;
	  el = i + (j-1)*E->lmesh.elz + (k-1)*E->lmesh.elz*E->lmesh.elx;
	  E->boundary.element[m][count] = el;
	  E->boundary.normal[m][dims][count] = 1;
	  for(d=1; d<dims; d++)
	    E->boundary.normal[m][d][count] = 0;
	  ++count;
	}

      } /* end for i, j, k */

    E->boundary.nel = count - 1;
  } /* end for m */
}
