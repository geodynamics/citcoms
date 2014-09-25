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

#include "global_defs.h"
#include "parallel_related.h"

void get_r_spacing_fine(double *,struct All_variables *);
void get_r_spacing_at_levels(double *,struct All_variables *);
 
#ifdef USE_GGRD
void ggrd_reg_temp_init(struct All_variables *);
#endif


/* =================================================
   Standard node positions including mesh refinement

   =================================================  */

void regional_node_locations(E)
  struct All_variables *E;
{
  int i,j,k,lev;
  double ro,dr,*rr,*RR,fo;
  float tt1;
  int nox,noy,noz,step;
  int nn;
  char output_file[255];
  char a[100];
  FILE *fp1;

  void regional_coord_of_cap();
  void compute_angle_surf_area ();
  void parallel_process_termination();
  void myerror();

  rr = (double *)  malloc((E->mesh.noz+1)*sizeof(double));
  RR = (double *)  malloc((E->mesh.noz+1)*sizeof(double));
  nox=E->mesh.nox;
  noy=E->mesh.noy;
  noz=E->mesh.noz;


  switch(E->control.coor)    {	
  case 0:
    /* default: regular node spacing */
    dr = (E->sphere.ro-E->sphere.ri)/(E->mesh.noz-1);
    for (k=1;k<=E->mesh.noz;k++)  {
      rr[k] = E->sphere.ri + (k-1)*dr;
    }
    break;
  case 1:
    /* get nodal levels from file */
    sprintf(output_file,"%s",E->control.coor_file);
    fp1=fopen(output_file,"r");
    if (fp1 == NULL) {
      fprintf(E->fp,"(Nodal_mesh.c #1) Cannot open %s\n",output_file);
      exit(8);
    }

    fscanf(fp1,"%s %d",a,&i);
    for(i=1;i<=nox;i++)
      fscanf(fp1,"%d %f",&nn,&tt1);

    fscanf(fp1,"%s %d",a,&i);
    for(i=1;i<=noy;i++)
      fscanf(fp1,"%d %f",&nn,&tt1);

    fscanf(fp1,"%s %d",a,&i);
    for (k=1;k<=E->mesh.noz;k++)  {
      if(fscanf(fp1,"%d %f",&nn,&tt1) != 2) {
        fprintf(stderr,"Error while reading file '%s'\n",output_file);
        exit(8);
      }
      rr[k]=tt1;
    }
    E->sphere.ri = rr[1];
    E->sphere.ro = rr[E->mesh.noz];

    fclose(fp1);
    break;
  case 2:
    /* higher radial spacing in top and bottom fractions */
    get_r_spacing_fine(rr, E);
    break;
   case 3:
     /*  assign radial spacing CitcomCU style */
     get_r_spacing_at_levels(rr,E);
    break;
 default:
    myerror(E,"regional_version_dependent: coor mode not implemented");
    break;
  }


  for (i=1;i<=E->mesh.noz;i++)  {
      E->sphere.gr[i] = rr[i];
      /* if(E->parallel.me==0) fprintf(stderr, "%d %f\n", i, E->sphere.gr[i]); */
  }

  for (i=1;i<=E->lmesh.noz;i++)  {
      k = E->lmesh.nzs+i-1;
      RR[i] = rr[k];
      }

  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++) {

    if (E->control.NMULTIGRID)
        step = (int) pow(2.0,(double)(E->mesh.levmax-lev));
    else
        step = 1;

      for (i=1;i<=E->lmesh.NOZ[lev];i++)
         E->sphere.R[lev][i] = RR[(i-1)*step+1];

    }          /* lev   */


     regional_coord_of_cap(E,0);


  if (E->control.verbose) {
  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++) {
    fprintf(E->fp_out,"output_coordinates before rotation %d \n",lev);
    fprintf(E->fp_out,"output_coordinates for cap %d %d\n",CPPR,E->lmesh.NNO[lev]);
    for (i=1;i<=E->lmesh.NNO[lev];i++)
      if(i%E->lmesh.NOZ[lev]==1)
        fprintf(E->fp_out,"%d %d %g %g %g\n",CPPR,i,E->SX[lev][1][i],E->SX[lev][2][i],E->SX[lev][3][i]);
    }
    fflush(E->fp_out);
  }

  compute_angle_surf_area (E);   /* used for interpolation */
#ifdef ALLOW_ELLIPTICAL
  if(E->data.use_ellipse)
    myerror("ellipticity not implemented for regional code",E);
#endif
  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)
      for (i=1;i<=E->lmesh.NNO[lev];i++)  {
        E->SinCos[lev][0][i] = sin(E->SX[lev][1][i]);
        E->SinCos[lev][1][i] = sin(E->SX[lev][2][i]);
        E->SinCos[lev][2][i] = cos(E->SX[lev][1][i]);
        E->SinCos[lev][3][i] = cos(E->SX[lev][2][i]);
      }

  if (E->control.verbose) {
  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)   {
    fprintf(E->fp_out,"output_coordinates after rotation %d \n",lev);
      for (i=1;i<=E->lmesh.NNO[lev];i++)
        if(i%E->lmesh.NOZ[lev]==1)
             fprintf(E->fp_out,"%d %d %g %g %g\n",CPPR,i,E->SX[lev][1][i],E->SX[lev][2][i],E->SX[lev][3][i]);
      }
    fflush(E->fp_out);
  }
   free((void *)rr);
   free((void *)RR);

   return;
}



/* setup boundary node and element arrays for bookkeeping */

void regional_construct_boundary( struct All_variables *E)
{
  const int dims=E->mesh.nsd;

  int m, i, j, k, d, el, count;
  int isBoundary;
  int normalFlag[4];

  /* boundary = all - interior */
  int max_size = E->lmesh.elx*E->lmesh.ely*E->lmesh.elz
    - (E->lmesh.elx-2)*(E->lmesh.ely-2)*(E->lmesh.elz-2) + 1;

  E->boundary.element = (int *)malloc(max_size*sizeof(int));

  for(d=1; d<=dims; d++)
    E->boundary.normal[CPPR][d] = (int *)malloc(max_size*sizeof(int));


    count = 1;
    for(k=1; k<=E->lmesh.ely; k++)
      for(j=1; j<=E->lmesh.elx; j++)
	for(i=1; i<=E->lmesh.elz; i++) {

	  isBoundary = 0;
	  for(d=1; d<=dims; d++)
	    normalFlag[d] = 0;

	  if((E->parallel.me_loc[1] == 0) && (j == 1)) {
	    isBoundary = 1;
	    normalFlag[1] = -1;
	  }

	  if((E->parallel.me_loc[1] == E->parallel.nprocx - 1)
	     && (j == E->lmesh.elx)) {
	    isBoundary = 1;
	    normalFlag[1] = 1;
	  }

	  if((E->parallel.me_loc[2] == 0) && (k == 1)) {
	    isBoundary = 1;
	    normalFlag[2] = -1;
	  }

	  if((E->parallel.me_loc[2] == E->parallel.nprocy - 1)
	     && (k == E->lmesh.ely)) {
	    isBoundary = 1;
	    normalFlag[2] = 1;
	  }

	  if((E->parallel.me_loc[3] == 0) && (i == 1)) {
	    isBoundary = 1;
	    normalFlag[3] = -1;
	  }

	  if((E->parallel.me_loc[3] == E->parallel.nprocz - 1)
	     && (i == E->lmesh.elz)) {
	    isBoundary = 1;
	    normalFlag[3] = 1;
	  }

	  if(isBoundary) {
	    el = i + (j-1)*E->lmesh.elz + (k-1)*E->lmesh.elz*E->lmesh.elx;
	    E->boundary.element[count] = el;
	    for(d=1; d<=dims; d++)
	      E->boundary.normal[CPPR][d][count] = normalFlag[d];

	    ++count;
	  }

	} /* end for i, j, k */

    E->boundary.nel = count - 1;
}
