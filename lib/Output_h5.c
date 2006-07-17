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
/* Routine to process the output of the finite element cycles as an HDF5 file */


#include <stdlib.h>
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "output_h5.h"

void h5output_coord(struct All_variables *);
void h5output_mat(struct All_variables *);
void h5output_velo(struct All_variables *, int);
void h5output_visc_prepare(struct All_variables *, float **);
void h5output_visc(struct All_variables *, int);
void h5output_surf_botm(struct All_variables *, int);
void h5output_surf_botm_pseudo_surf(struct All_variables *, int);
void h5output_stress(struct All_variables *, int);
void h5output_ave_r(struct All_variables *, int);
void h5output_tracer(struct All_variables *, int);

extern void parallel_process_termination();
extern void heat_flux(struct All_variables *);
extern void get_STD_topo(struct All_variables *, float**, float**,
			 float**, float**, int);

/**********************************************************************/

void h5output_open(struct All_variables *E)
{
  FILE *fp1;

  // if filename is empty, h5output to stderr.
  if (*filename) {
    fp1 = fopen(filename,"w");
    if (!fp1) {
      fprintf(stderr,"Cannot open file '%s'\n",filename);
      parallel_process_termination();
    }
  }
  else
    fp1 = stderr;

  //return fp1;
  return;
}

void h5output_close(struct All_variables *E)
{
    return;
}


void h5output(struct All_variables *E, int cycles)
{

  if (cycles == 0) {
    h5output_coord(E);
    h5output_mat(E);
  }

  h5output_velo(E, cycles);
  h5output_visc(E, cycles);

  if(E->control.pseudo_free_surf) {
    if(E->mesh.topvbc == 2)
       h5output_surf_botm_pseudo_surf(E, cycles);
  }
  else
    h5output_surf_botm(E, cycles);

  if(E->control.tracer==1)
    h5output_tracer(E, cycles);

  //h5output_stress(E, cycles);
  //h5output_pressure(E, cycles);

  /* disable horizontal average h5output   by Tan2 */
  /* h5output_ave_r(E, cycles); */

  return;
}


void h5output_pseudo_surf(struct All_variables *E, int cycles)
{

  if (cycles == 0) {
    h5output_coord(E);
    h5output_mat(E);
  }

  h5output_velo(E, cycles);
  h5output_visc(E, cycles);
  h5output_surf_botm_pseudo_surf(E, cycles);

  if(E->control.tracer==1)
    h5output_tracer(E, cycles);

  //h5output_stress(E, cycles);
  //h5output_pressure(E, cycles);

  /* disable horizontal average h5output   by Tan2 */
  /* h5output_ave_r(E, cycles); */

  return;
}


void h5output_coord(struct All_variables *E)
{
  int i, j;
  char h5output_file[255];
  FILE *fp1;

  sprintf(h5output_file,"%s.coord.%d",E->control.data_file,E->parallel.me);
  fp1 = h5output_open(h5output_file);

  for(j=1;j<=E->sphere.caps_per_proc;j++)     {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.6e %.6e %.6e\n",E->sx[j][1][i],E->sx[j][2][i],E->sx[j][3][i]);
  }

  fclose(fp1);

  return;
}


void h5output_visc(struct All_variables *E, int cycles)
{
  int i, j;
  char h5output_file[255];
  FILE *fp1;
  int lev = E->mesh.levmax;

  sprintf(h5output_file,"%s.visc.%d.%d",E->control.data_file,E->parallel.me,cycles);
  fp1 = h5output_open(h5output_file);


  for(j=1;j<=E->sphere.caps_per_proc;j++) {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.4e\n",E->VI[lev][j][i]);
  }

  fclose(fp1);

  return;
}


void h5output_velo(struct All_variables *E, int cycles)
{
  int i, j;
  char h5output_file[255];
  FILE *fp1;

  sprintf(h5output_file,"%s.velo.%d.%d",E->control.data_file,E->parallel.me,cycles);
  fp1 = h5output_open(h5output_file);

  fprintf(fp1,"%d %d %.5e\n",cycles,E->lmesh.nno,E->monitor.elapsed_time);

  for(j=1;j<=E->sphere.caps_per_proc;j++) {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++) {
      fprintf(fp1,"%.6e %.6e %.6e %.6e\n",E->sphere.cap[j].V[1][i],E->sphere.cap[j].V[2][i],E->sphere.cap[j].V[3][i],E->T[j][i]);
    }
  }

  fclose(fp1);

  return;
}


void h5output_surf_botm(struct All_variables *E, int cycles)
{
  int i, j, s;
  char h5output_file[255];
  FILE* fp2;


  heat_flux(E);
  get_STD_topo(E,E->slice.tpg,E->slice.tpgb,E->slice.divg,E->slice.vort,cycles);

  if (E->parallel.me_loc[3]==E->parallel.nprocz-1) {
    sprintf(h5output_file,"%s.surf.%d.%d",E->control.data_file,E->parallel.me,cycles);
    fp2 = h5output_open(h5output_file);

    for(j=1;j<=E->sphere.caps_per_proc;j++)  {
	    fprintf(fp2,"%3d %7d\n",j,E->lmesh.nsf);
	    for(i=1;i<=E->lmesh.nsf;i++)   {
		    s = i*E->lmesh.noz;
		    fprintf(fp2,"%.4e %.4e %.4e %.4e\n",E->slice.tpg[j][i],E->slice.shflux[j][i],E->sphere.cap[j].V[1][s],E->sphere.cap[j].V[2][s]);
	    }
    }
    fclose(fp2);
  }


  if (E->parallel.me_loc[3]==0)      {
    sprintf(h5output_file,"%s.botm.%d.%d",E->control.data_file,E->parallel.me,cycles);
    fp2 = h5output_open(h5output_file);

    for(j=1;j<=E->sphere.caps_per_proc;j++)  {
      fprintf(fp2,"%3d %7d\n",j,E->lmesh.nsf);
      for(i=1;i<=E->lmesh.nsf;i++)  {
	s = (i-1)*E->lmesh.noz + 1;
        fprintf(fp2,"%.4e %.4e %.4e %.4e\n",E->slice.tpgb[j][i],E->slice.bhflux[j][i],E->sphere.cap[j].V[1][s],E->sphere.cap[j].V[2][s]);
      }
    }
    fclose(fp2);
  }

  return;
}

void h5output_surf_botm_pseudo_surf(struct All_variables *E, int cycles)
{
  int i, j, s;
  char h5output_file[255];
  FILE* fp2;


  heat_flux(E);
  get_STD_topo(E,E->slice.tpg,E->slice.tpgb,E->slice.divg,E->slice.vort,cycles);

  if (E->parallel.me_loc[3]==E->parallel.nprocz-1) {
    sprintf(h5output_file,"%s.surf.%d.%d",E->control.data_file,E->parallel.me,cycles);
    fp2 = h5output_open(h5output_file);

    for(j=1;j<=E->sphere.caps_per_proc;j++)  {
	    fprintf(fp2,"%3d %7d\n",j,E->lmesh.nsf);
	    for(i=1;i<=E->lmesh.nsf;i++)   {
		    s = i*E->lmesh.noz;
		    fprintf(fp2,"%.4e %.4e %.4e %.4e\n",E->slice.freesurf[j][i],E->slice.shflux[j][i],E->sphere.cap[j].V[1][s],E->sphere.cap[j].V[2][s]);
	    }
    }
    fclose(fp2);
  }


  if (E->parallel.me_loc[3]==0)      {
    sprintf(h5output_file,"%s.botm.%d.%d",E->control.data_file,E->parallel.me,cycles);
    fp2 = h5output_open(h5output_file);

    for(j=1;j<=E->sphere.caps_per_proc;j++)  {
      fprintf(fp2,"%3d %7d\n",j,E->lmesh.nsf);
      for(i=1;i<=E->lmesh.nsf;i++)  {
	s = (i-1)*E->lmesh.noz + 1;
	fprintf(fp2,"%.4e %.4e %.4e %.4e\n",E->slice.tpgb[j][i],E->slice.bhflux[j][i],E->sphere.cap[j].V[1][s],E->sphere.cap[j].V[2][s]);
      }
    }
    fclose(fp2);
  }

  return;
}


void h5output_stress(struct All_variables *E, int cycles)
{
  int m, node;
  char h5output_file[255];
  FILE *fp1;

  sprintf(h5output_file,"%s.stress.%d.%d",E->control.data_file,E->parallel.me,cycles);
  fp1 = h5output_open(h5output_file);

  fprintf(fp1,"%d %d %.5e\n",cycles,E->lmesh.nno,E->monitor.elapsed_time);

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    fprintf(fp1,"%3d %7d\n",m,E->lmesh.nno);
    for (node=1;node<=E->lmesh.nno;node++)
      fprintf(fp1, "%d %e %e %e %e %e %e\n", node,
	      E->gstress[m][(node-1)*6+1],
	      E->gstress[m][(node-1)*6+2],
	      E->gstress[m][(node-1)*6+3],
	      E->gstress[m][(node-1)*6+4],
	      E->gstress[m][(node-1)*6+5],
	      E->gstress[m][(node-1)*6+6]);
  }
  fclose(fp1);
}


void h5output_avg_r(struct All_variables *E, int cycles)
{
  /* horizontal average h5output of temperature and rms velocity*/
  void return_horiz_ave_f();

  int m, i, j;
  float *S1[NCS],*S2[NCS],*S3[NCS];
  char h5output_file[255];
  FILE *fp1;

  // compute horizontal average here....

  for(m=1;m<=E->sphere.caps_per_proc;m++)      {
    S1[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
    S2[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
    S3[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
  }

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    for(i=1;i<=E->lmesh.nno;i++) {
      S1[m][i] = E->T[m][i];
      S2[m][i] = E->sphere.cap[m].V[1][i]*E->sphere.cap[m].V[1][i]
          	+ E->sphere.cap[m].V[2][i]*E->sphere.cap[m].V[2][i];
      S3[m][i] = E->sphere.cap[m].V[3][i]*E->sphere.cap[m].V[3][i];
    }
  }

  return_horiz_ave_f(E,S1,E->Have.T);
  return_horiz_ave_f(E,S2,E->Have.V[1]);
  return_horiz_ave_f(E,S3,E->Have.V[2]);

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    free((void *)S1[m]);
    free((void *)S2[m]);
    free((void *)S3[m]);
  }

  for (i=1;i<=E->lmesh.noz;i++) {
      E->Have.V[1][i] = sqrt(E->Have.V[1][i]);
      E->Have.V[2][i] = sqrt(E->Have.V[2][i]);
  }

  // only the first nprocz processors need to h5output

  if (E->parallel.me<E->parallel.nprocz)  {
    sprintf(h5output_file,"%s.ave_r.%d.%d",E->control.data_file,E->parallel.me,cycles);
    fp1=fopen(h5output_file,"w");
    for(j=1;j<=E->lmesh.noz;j++)  {
        fprintf(fp1,"%.4e %.4e %.4e %.4e\n",E->sx[1][3][j],E->Have.T[j],E->Have.V[1][j],E->Have.V[2][j]);
    }
    fclose(fp1);
  }

  return;
}



void h5output_mat(struct All_variables *E)
{
  int m, el;
  char h5output_file[255];
  FILE* fp;

  sprintf(h5output_file,"%s.mat.%d",E->control.data_file,E->parallel.me);
  fp = h5output_open(h5output_file);

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(el=1;el<=E->lmesh.nel;el++)
      fprintf(fp,"%d %d %f\n", el,E->mat[m][el],E->VIP[m][el]);

  fclose(fp);

  return;
}



void h5output_pressure(struct All_variables *E, int cycles)
{
  int i, j;
  char h5output_file[255];
  FILE *fp1;

  sprintf(h5output_file,"%s.pressure.%d.%d",E->control.data_file,E->parallel.me,cycles);
  fp1 = h5output_open(h5output_file);

  fprintf(fp1,"%d %d %.5e\n",cycles,E->lmesh.nno,E->monitor.elapsed_time);

  for(j=1;j<=E->sphere.caps_per_proc;j++) {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.6e\n",E->NP[j][i]);
  }

  fclose(fp1);

  return;
}



void h5output_tracer(struct All_variables *E, int cycles)
{
  int n;
  char h5output_file[255];
  FILE *fp1;

  sprintf(h5output_file,"%s.tracer.%d.%d",E->control.data_file,E->parallel.me,cycles);
  fp1 = h5output_open(h5output_file);

  fprintf(fp1,"%.5e\n",E->monitor.elapsed_time);

  for(n=1;n<=E->Tracer.LOCAL_NUM_TRACERS;n++)   {
    fprintf(fp1,"%.4e %.4e %.4e %.4e\n", E->Tracer.itcolor[n], E->Tracer.tracer_x[n],E->Tracer.tracer_y[n],E->Tracer.tracer_z[n]);
  }

  fclose(fp1);

  return;
}


