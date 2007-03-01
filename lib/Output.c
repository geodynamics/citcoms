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
/* Routine to process the output of the finite element cycles
   and to turn them into a coherent suite  files  */


#include <stdlib.h>
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "parsing.h"
#include "output.h"

void output_coord(struct All_variables *);
void output_mat(struct All_variables *);
void output_velo(struct All_variables *, int);
void output_visc_prepare(struct All_variables *, float **);
void output_visc(struct All_variables *, int);
void output_surf_botm(struct All_variables *, int);
void output_geoid(struct All_variables *, int);
void output_stress(struct All_variables *, int);
void output_horiz_avg(struct All_variables *, int);
void output_tracer(struct All_variables *, int);
void output_pressure(struct All_variables *, int);

extern void parallel_process_termination();
extern void heat_flux(struct All_variables *);
extern void get_STD_topo(struct All_variables *, float**, float**,
                         float**, float**, int);

/**********************************************************************/

void output_common_input(struct All_variables *E)
{
    int m = E->parallel.me;

    input_string("output_format", E->output.format, "ascii",m);
    input_string("output_optional", E->output.optional, "surf,botm",m);

}



void output(struct All_variables *E, int cycles)
{

  if (cycles == 0) {
    output_coord(E);
    /*output_mat(E);*/
  }

  output_velo(E, cycles);
  output_visc(E, cycles);

  output_surf_botm(E, cycles);

  /* output tracer location if using tracer */
  //TODO: output_tracer() is not working for full version of tracers
  //if(E->control.tracer==1)
  //output_tracer(E, cycles);

  /* optiotnal output below */
  /* compute and output geoid (in spherical harmonics coeff) */
  if (E->output.geoid == 1)
      output_geoid(E, cycles);

  if (E->output.stress == 1)
    output_stress(E, cycles);

  if (E->output.pressure == 1)
    output_pressure(E, cycles);

  if (E->output.horiz_avg == 1)
      output_horiz_avg(E, cycles);

  return;
}


FILE* output_open(char *filename)
{
  FILE *fp1;

  /* if filename is empty, output to stderr. */
  if (*filename) {
    fp1 = fopen(filename,"w");
    if (!fp1) {
      fprintf(stderr,"Cannot open file '%s'\n",filename);
      parallel_process_termination();
    }
  }
  else
    fp1 = stderr;

  return fp1;
}


void output_coord(struct All_variables *E)
{
  int i, j;
  char output_file[255];
  FILE *fp1;

  sprintf(output_file,"%s.coord.%d",E->control.data_file,E->parallel.me);
  fp1 = output_open(output_file);

  for(j=1;j<=E->sphere.caps_per_proc;j++)     {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.6e %.6e %.6e\n",E->sx[j][1][i],E->sx[j][2][i],E->sx[j][3][i]);
  }

  fclose(fp1);

  return;
}


void output_visc(struct All_variables *E, int cycles)
{
  int i, j;
  char output_file[255];
  FILE *fp1;
  int lev = E->mesh.levmax;

  sprintf(output_file,"%s.visc.%d.%d", E->control.data_file,
          E->parallel.me, cycles);
  fp1 = output_open(output_file);


  for(j=1;j<=E->sphere.caps_per_proc;j++) {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.4e\n",E->VI[lev][j][i]);
  }

  fclose(fp1);

  return;
}


void output_velo(struct All_variables *E, int cycles)
{
  int i, j;
  char output_file[255];
  FILE *fp1;

  sprintf(output_file,"%s.velo.%d.%d", E->control.data_file,
          E->parallel.me, cycles);
  fp1 = output_open(output_file);

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


void output_surf_botm(struct All_variables *E, int cycles)
{
  int i, j, s;
  char output_file[255];
  FILE* fp2;
  float *topo;

  heat_flux(E);
  get_STD_topo(E,E->slice.tpg,E->slice.tpgb,E->slice.divg,E->slice.vort,cycles);

  if (E->output.surf && (E->parallel.me_loc[3]==E->parallel.nprocz-1)) {
    sprintf(output_file,"%s.surf.%d.%d", E->control.data_file,
            E->parallel.me, cycles);
    fp2 = output_open(output_file);

    for(j=1;j<=E->sphere.caps_per_proc;j++)  {
        /* choose either STD topo or pseudo-free-surf topo */
        if(E->control.pseudo_free_surf)
            topo = E->slice.freesurf[j];
        else
            topo = E->slice.tpg[j];

        fprintf(fp2,"%3d %7d\n",j,E->lmesh.nsf);
        for(i=1;i<=E->lmesh.nsf;i++)   {
            s = i*E->lmesh.noz;
            fprintf(fp2,"%.4e %.4e %.4e %.4e\n",topo[i],E->slice.shflux[j][i],E->sphere.cap[j].V[1][s],E->sphere.cap[j].V[2][s]);
        }
    }
    fclose(fp2);
  }


  if (E->output.botm && (E->parallel.me_loc[3]==0)) {
    sprintf(output_file,"%s.botm.%d.%d", E->control.data_file,
            E->parallel.me, cycles);
    fp2 = output_open(output_file);

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


void output_geoid(struct All_variables *E, int cycles)
{
    void compute_geoid();
    int ll, mm, p;
    char output_file[255];
    FILE *fp1;

    compute_geoid(E, E->sphere.harm_geoid,
                  E->sphere.harm_geoid_from_bncy,
                  E->sphere.harm_geoid_from_tpgt,
                  E->sphere.harm_geoid_from_tpgb);

    if (E->parallel.me == (E->parallel.nprocz-1))  {
        sprintf(output_file, "%s.geoid.%d.%d", E->control.data_file,
                E->parallel.me, cycles);
        fp1 = output_open(output_file);

        /* write headers */
        fprintf(fp1, "%d %d %.5e\n", cycles, E->output.llmax,
                E->monitor.elapsed_time);

        /* write sph harm coeff of geoid and topos */
        for (ll=0; ll<=E->output.llmax; ll++)
            for(mm=0; mm<=ll; mm++)  {
                p = E->sphere.hindex[ll][mm];
                fprintf(fp1,"%d %d %.4e %.4e %.4e %.4e %.4e %.4e\n",
                        ll, mm,
                        E->sphere.harm_geoid[0][p],
                        E->sphere.harm_geoid[1][p],
                        E->sphere.harm_geoid_from_tpgt[0][p],
                        E->sphere.harm_geoid_from_tpgt[1][p],
                        E->sphere.harm_geoid_from_bncy[0][p],
                        E->sphere.harm_geoid_from_bncy[1][p]);

            }

        fclose(fp1);
    }
}



void output_stress(struct All_variables *E, int cycles)
{
  int m, node;
  char output_file[255];
  FILE *fp1;

  sprintf(output_file,"%s.stress.%d.%d", E->control.data_file,
          E->parallel.me, cycles);
  fp1 = output_open(output_file);

  fprintf(fp1,"%d %d %.5e\n",cycles,E->lmesh.nno,E->monitor.elapsed_time);

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    fprintf(fp1,"%3d %7d\n",m,E->lmesh.nno);
    for (node=1;node<=E->lmesh.nno;node++)
      fprintf(fp1, "%.4e %.4e %.4e %.4e %.4e %.4e\n",
              E->gstress[m][(node-1)*6+1],
              E->gstress[m][(node-1)*6+2],
              E->gstress[m][(node-1)*6+3],
              E->gstress[m][(node-1)*6+4],
              E->gstress[m][(node-1)*6+5],
              E->gstress[m][(node-1)*6+6]);
  }
  fclose(fp1);
}


void output_horiz_avg(struct All_variables *E, int cycles)
{
  /* horizontal average output of temperature and rms velocity*/
  void compute_horiz_avg();

  int j;
  char output_file[255];
  FILE *fp1;

  /* compute horizontal average here.... */
  compute_horiz_avg(E);

  /* only the first nprocz processors need to output */

  if (E->parallel.me<E->parallel.nprocz)  {
    sprintf(output_file,"%s.horiz_avg.%d.%d", E->control.data_file,
            E->parallel.me, cycles);
    fp1=fopen(output_file,"w");
    for(j=1;j<=E->lmesh.noz;j++)  {
        fprintf(fp1,"%.4e %.4e %.4e %.4e\n",E->sx[1][3][j],E->Have.T[j],E->Have.V[1][j],E->Have.V[2][j]);
    }
    fclose(fp1);
  }

  return;
}



void output_mat(struct All_variables *E)
{
  int m, el;
  char output_file[255];
  FILE* fp;

  sprintf(output_file,"%s.mat.%d", E->control.data_file,E->parallel.me);
  fp = output_open(output_file);

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(el=1;el<=E->lmesh.nel;el++)
      fprintf(fp,"%d %d %f\n", el,E->mat[m][el],E->VIP[m][el]);

  fclose(fp);

  return;
}



void output_pressure(struct All_variables *E, int cycles)
{
  int i, j;
  char output_file[255];
  FILE *fp1;

  sprintf(output_file,"%s.pressure.%d.%d", E->control.data_file,
          E->parallel.me, cycles);
  fp1 = output_open(output_file);

  fprintf(fp1,"%d %d %.5e\n",cycles,E->lmesh.nno,E->monitor.elapsed_time);

  for(j=1;j<=E->sphere.caps_per_proc;j++) {
    fprintf(fp1,"%3d %7d\n",j,E->lmesh.nno);
    for(i=1;i<=E->lmesh.nno;i++)
      fprintf(fp1,"%.6e\n",E->NP[j][i]);
  }

  fclose(fp1);

  return;
}



void output_tracer(struct All_variables *E, int cycles)
{
  int n;
  char output_file[255];
  FILE *fp1;

  sprintf(output_file,"%s.tracer.%d.%d", E->control.data_file,
          E->parallel.me, cycles);
  fp1 = output_open(output_file);

  fprintf(fp1,"%.5e\n",E->monitor.elapsed_time);

  for(n=1;n<=E->Tracer.LOCAL_NUM_TRACERS;n++)   {
    fprintf(fp1,"%.4e %.4e %.4e %.4e\n", E->Tracer.itcolor[n], E->Tracer.tracer_x[n],E->Tracer.tracer_y[n],E->Tracer.tracer_z[n]);
  }

  fclose(fp1);

  return;
}


void output_time(struct All_variables *E, int cycles)
{
  double CPU_time0();

  double current_time = CPU_time0();

  if (E->parallel.me == 0) {
    fprintf(E->fptime,"%d %.4e %.4e %.4e %.4e\n",
            cycles,
            E->monitor.elapsed_time,
            E->advection.timestep,
            current_time - E->monitor.cpu_time_at_start,
            current_time - E->monitor.cpu_time_at_last_cycle);

    fflush(E->fptime);
  }

  E->monitor.cpu_time_at_last_cycle = current_time;

  return;
}

