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

void output_comp_nd(struct All_variables *, int);
void output_comp_el(struct All_variables *, int);
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
void output_heating(struct All_variables *, int);

extern void parallel_process_termination();
extern void heat_flux(struct All_variables *);
extern void get_STD_topo(struct All_variables *, float**, float**,
                         float**, float**, int);

/**********************************************************************/

void output_common_input(struct All_variables *E)
{
    int m = E->parallel.me;

    input_string("output_format", E->output.format, "ascii",m);
    input_string("output_optional", E->output.optional, "surf,botm,tracer",m);

    /* gzdir type of I/O */
    E->output.gzdir.vtk_io = 0;
    E->output.gzdir.rnr = 0;
    if(strcmp(E->output.format, "ascii-gz") == 0){
      /*
	 vtk_io = 1: write files for post-processing into VTK
	 vtk_io = 2: write serial legacy VTK file
	 vtk_io = 3: write paralle legacy VTK file

      */
      input_int("gzdir_vtkio",&(E->output.gzdir.vtk_io),"0",m);
      /* remove net rotation on output step? */
      input_boolean("gzdir_rnr",&(E->output.gzdir.rnr),"off",m);
      E->output.gzdir.vtk_base_init = 0;
      E->output.gzdir.vtk_base_save = 1; /* should we save the basis vectors? (memory!) */
      //fprintf(stderr,"gzdir: vtkio: %i save basis vectors: %i\n",
      //      E->output.gzdir.vtk_io,E->output.gzdir.vtk_base_save);
    }
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

  /* optiotnal output below */

  /* compute and output geoid (in spherical harmonics coeff) */
  if (E->output.geoid)
      output_geoid(E, cycles);

  if (E->output.stress)
    output_stress(E, cycles);

  if (E->output.pressure)
    output_pressure(E, cycles);

  if (E->output.horiz_avg)
      output_horiz_avg(E, cycles);

  if(E->output.tracer && E->control.tracer)
      output_tracer(E, cycles);

  if (E->output.comp_nd && E->composition.on)
      output_comp_nd(E, cycles);

  if (E->output.comp_el && E->composition.on)
      output_comp_el(E, cycles);

  if(E->output.heating && E->control.disptn_number != 0)
      output_heating(E, cycles);

  return;
}


FILE* output_open(char *filename, char *mode)
{
  FILE *fp1;

  /* if filename is empty, output to stderr. */
  if (*filename) {
    fp1 = fopen(filename,mode);
    if (!fp1) {
      fprintf(stderr,"Cannot open file '%s' for '%s'\n",
	      filename,mode);
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
  fp1 = output_open(output_file, "w");

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
  fp1 = output_open(output_file, "w");


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
  fp1 = output_open(output_file, "w");

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

  if((E->output.write_q_files == 0) || (cycles == 0) ||
     (cycles % E->output.write_q_files)!=0)
      heat_flux(E);
  /* else, the heat flux will have been computed already */

  get_STD_topo(E,E->slice.tpg,E->slice.tpgb,E->slice.divg,E->slice.vort,cycles);

  if (E->output.surf && (E->parallel.me_loc[3]==E->parallel.nprocz-1)) {
    sprintf(output_file,"%s.surf.%d.%d", E->control.data_file,
            E->parallel.me, cycles);
    fp2 = output_open(output_file, "w");

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
    fp2 = output_open(output_file, "w");

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
        fp1 = output_open(output_file, "w");

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
  fp1 = output_open(output_file, "w");

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
  fp = output_open(output_file, "w");

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
  fp1 = output_open(output_file, "w");

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
  int i, j, n, ncolumns;
  char output_file[255];
  FILE *fp1;

  sprintf(output_file,"%s.tracer.%d.%d", E->control.data_file,
          E->parallel.me, cycles);
  fp1 = output_open(output_file, "w");

  ncolumns = 3 + E->trace.number_of_extra_quantities;

  for(j=1;j<=E->sphere.caps_per_proc;j++) {
      fprintf(fp1,"%d %d %d %.5e\n", cycles, E->trace.ntracers[j],
              ncolumns, E->monitor.elapsed_time);

      for(n=1;n<=E->trace.ntracers[j];n++) {
          /* write basic quantities (coordinate) */
          fprintf(fp1,"%.12e %.12e %.12e",
                  E->trace.basicq[j][0][n],
                  E->trace.basicq[j][1][n],
                  E->trace.basicq[j][2][n]);

          /* write extra quantities */
          for (i=0; i<E->trace.number_of_extra_quantities; i++) {
              fprintf(fp1," %.12e", E->trace.extraq[j][i][n]);
          }
          fprintf(fp1, "\n");
      }

  }

  fclose(fp1);
  return;
}


void output_comp_nd(struct All_variables *E, int cycles)
{
    int i, j, k;
    char output_file[255];
    FILE *fp1;

    sprintf(output_file,"%s.comp_nd.%d.%d", E->control.data_file,
            E->parallel.me, cycles);
    fp1 = output_open(output_file, "w");

    for(j=1;j<=E->sphere.caps_per_proc;j++) {
        fprintf(fp1,"%3d %7d %.5e %d\n",
                j, E->lmesh.nel,
                E->monitor.elapsed_time, E->composition.ncomp);
        for(i=0;i<E->composition.ncomp;i++) {
            fprintf(fp1,"%.5e %.5e ",
                    E->composition.initial_bulk_composition[i],
                    E->composition.bulk_composition[i]);
        }
        fprintf(fp1,"\n");

        for(i=1;i<=E->lmesh.nno;i++) {
            for(k=0;k<E->composition.ncomp;k++) {
                fprintf(fp1,"%.6e ",E->composition.comp_node[j][k][i]);
            }
            fprintf(fp1,"\n");
        }

    }

    fclose(fp1);
    return;
}


void output_comp_el(struct All_variables *E, int cycles)
{
    int i, j, k;
    char output_file[255];
    FILE *fp1;

    sprintf(output_file,"%s.comp_el.%d.%d", E->control.data_file,
            E->parallel.me, cycles);
    fp1 = output_open(output_file, "w");

    for(j=1;j<=E->sphere.caps_per_proc;j++) {
        fprintf(fp1,"%3d %7d %.5e %d\n",
                j, E->lmesh.nel,
                E->monitor.elapsed_time, E->composition.ncomp);
        for(i=0;i<E->composition.ncomp;i++) {
            fprintf(fp1,"%.5e %.5e ",
                    E->composition.initial_bulk_composition[i],
                    E->composition.bulk_composition[i]);
        }
        fprintf(fp1,"\n");

        for(i=1;i<=E->lmesh.nno;i++) {
            for(k=0;k<E->composition.ncomp;k++) {
                fprintf(fp1,"%.6e ",
			E->composition.comp_el[j][k][i]);
            }
            fprintf(fp1,"\n");
        }
    }

    fclose(fp1);
    return;
}


void output_heating(struct All_variables *E, int cycles)
{
    int j, e;
    char output_file[255];
    FILE *fp1;

    sprintf(output_file,"%s.heating.%d.%d", E->control.data_file,
            E->parallel.me, cycles);
    fp1 = output_open(output_file, "w");

    fprintf(fp1,"%.5e\n",E->monitor.elapsed_time);

    for(j=1;j<=E->sphere.caps_per_proc;j++) {
        fprintf(fp1,"%3d %7d\n", j, E->lmesh.nel);
        for(e=1; e<=E->lmesh.nel; e++)
            fprintf(fp1, "%.4e %.4e %.4e\n", E->heating_adi[j][e],
                    E->heating_visc[j][e], E->heating_latent[j][e]);
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
