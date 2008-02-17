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
#include <assert.h>

#include "global_defs.h"
#include "lith_age.h"
#include "parsing.h"

void parallel_process_termination();
void temperatures_conform_bcs();

#include "initial_temperature.h"
void debug_tic(struct All_variables *);
void read_tic_from_file(struct All_variables *);

#ifdef USE_GZDIR
void restart_tic_from_gzdir_file(struct All_variables *);
#endif


void tic_input(struct All_variables *E)
{

  int m = E->parallel.me;
  int noz = E->lmesh.noz;
  int n;


  input_int("tic_method", &(E->convection.tic_method), "0,0,2", m);
  /* When tic_method is 0 (default), the temperature is a linear profile +
     perturbation at some layers.

     When tic_method is -1, the temperature is read in from the
     [datafile_old].velo.[rank].[solution_cycles_init] files.

     When tic_method is 1, the temperature is isothermal (== bottom b.c.) +
     uniformly cold plate (thickness specified by 'half_space_age').

     When tic_method is 2, (tic_method==1) + a hot blob. A user can specify
     the location and radius of the blob, and also the amplitude of temperature
     change in the blob relative to the ambient mantle temperautre
     (E->control.lith_age_mantle_temp).
        - blob_center: A comma-separated list of three float numbers.
        - blob_radius: A dmensionless length, typically a fraction
                       of the Earth's radius.
        - blob_dT    : Dimensionless temperature.

     When tic_method is 3, the temperature is a linear profile + perturbation
     for whole mantle.

     tic_method is 4: read in initial temperature distribution from a set of netcdf grd
                      files. this required the GGRD extension to be compiled in

  */

  switch(E->convection.tic_method){
  case -1:	/* read from file, no other options needed */
    break;
  case 0:
  case 3:
    /* This part put a temperature anomaly at depth where the global
       node number is equal to load_depth. The horizontal pattern of
       the anomaly is given by spherical harmonic ll & mm. */

    input_int("num_perturbations", &n, "0,0,PERTURB_MAX_LAYERS", m);

    if (n > 0) {
      E->convection.number_of_perturbations = n;

      if (! input_float_vector("perturbmag", n, E->convection.perturb_mag, m) ) {
	fprintf(stderr,"Missing input parameter: 'perturbmag'\n");
	parallel_process_termination();
      }
      if (! input_int_vector("perturbm", n, E->convection.perturb_mm, m) ) {
	fprintf(stderr,"Missing input parameter: 'perturbm'\n");
	parallel_process_termination();
      }
      if (! input_int_vector("perturbl", n, E->convection.perturb_ll, m) ) {
	fprintf(stderr,"Missing input parameter: 'perturbl'\n");
	parallel_process_termination();
      }
      if (! input_int_vector("perturblayer", n, E->convection.load_depth, m) ) {
	fprintf(stderr,"Missing input parameter: 'perturblayer'\n");
	parallel_process_termination();
      }
    }
    else {
      E->convection.number_of_perturbations = 1;
      E->convection.perturb_mag[0] = 1;
      E->convection.perturb_mm[0] = 2;
      E->convection.perturb_ll[0] = 2;
      E->convection.load_depth[0] = (noz+1)/2;
    }

    break;
  case 1:			/* case 1 */

    input_float("half_space_age", &(E->convection.half_space_age), "40.0,1e-3,nomax", m);
    break;

  case 2:			/* case 2 */
    input_float("half_space_age", &(E->convection.half_space_age), "40.0,1e-3,nomax", m);
    if( ! input_float_vector("blob_center", 3, E->convection.blob_center, m)) {
      assert( E->sphere.caps == 12 || E->sphere.caps == 1 );
      if(E->sphere.caps == 12) { /* Full version: just quit here */
        fprintf(stderr,"Missing input parameter: 'blob_center'.\n");
        parallel_process_termination();
      }
      else if(E->sphere.caps == 1) { /* Regional version: put the blob at the center */
        fprintf(stderr,"Missing input parameter: 'blob_center'. The blob will be placed at the center of the domain.\n");
        E->convection.blob_center[0] = 0.5*(E->control.theta_min+E->control.theta_max);
        E->convection.blob_center[1] = 0.5*(E->control.fi_min+E->control.fi_max);
        E->convection.blob_center[2] = 0.5*(E->sphere.ri+E->sphere.ro);
      }
    }
    input_float("blob_radius", &(E->convection.blob_radius), "0.063,0.0,1.0", m);
    input_float("blob_dT", &(E->convection.blob_dT), "0.18,nomin,nomax", m);
    break;
  case 4:
    /*
       case 4: initial temp from grd files
    */
#ifdef USE_GGRD
    /* read in some more parameters */
    /* scale the anomalies with PREM densities */
    input_boolean("ggrd_tinit_scale_with_prem",&(E->control.ggrd.temp_init.scale_with_prem),"off",E->parallel.me);
    /* limit T to 0...1 */
    input_boolean("ggrd_tinit_limit_trange",&(E->control.ggrd.temp_init.limit_trange),"on",E->parallel.me);
   /* scaling factor for the grids */
    input_double("ggrd_tinit_scale",&(E->control.ggrd.temp_init.scale),"1.0",E->parallel.me); /* scale */
    /* temperature offset factor */
    input_double("ggrd_tinit_offset",&(E->control.ggrd.temp_init.offset),"0.0",E->parallel.me); /* offset */
    /* grid name, without the .i.grd suffix */
    input_string("ggrd_tinit_gfile",E->control.ggrd.temp_init.gfile,"",E->parallel.me); /* grids */
    input_string("ggrd_tinit_dfile",E->control.ggrd.temp_init.dfile,"",E->parallel.me); /* depth.dat layers of grids*/
    /* override temperature boundary condition? */
    input_boolean("ggrd_tinit_override_tbc",&(E->control.ggrd.temp_init.override_tbc),"off",E->parallel.me);
    input_string("ggrd_tinit_prem_file",E->control.ggrd.temp_init.prem.model_filename,"", E->parallel.me); /* PREM model filename */
#else
    fprintf(stderr,"tic_method 4 only works for USE_GGRD compiled code\n");
    parallel_process_termination();
#endif

    break;
  default:			/* unknown option */
    fprintf(stderr,"Invalid value of 'tic_method'\n");
    parallel_process_termination();
    break;
  }

  return;
}



void convection_initial_temperature(struct All_variables *E)
{
  void report();

  report(E,"Initialize temperature field");

  if (E->convection.tic_method == -1) {
      /* read temperature from file */
#ifdef USE_GZDIR
      if(strcmp(E->output.format, "ascii-gz") == 0)
          restart_tic_from_gzdir_file(E);
      else
#endif
          read_tic_from_file(E);
  }
  else if (E->control.lith_age)
    lith_age_construct_tic(E);
  else
    (E->solver.construct_tic_from_input)(E);

  /* Note: it is the callee's responsibility to conform tbc. */
  /* like a call to temperatures_conform_bcs(E); */

  if (E->control.verbose)
    debug_tic(E);

  return;
}


void debug_tic(struct All_variables *E)
{
  int m, j;

  fprintf(E->fp_out,"output_temperature\n");
  for(m=1;m<=E->sphere.caps_per_proc;m++)        {
    fprintf(E->fp_out,"for cap %d\n",E->sphere.capid[m]);
    for (j=1;j<=E->lmesh.nno;j++)
      fprintf(E->fp_out,"X = %.6e Z = %.6e Y = %.6e T[%06d] = %.6e \n",E->sx[m][1][j],E->sx[m][2][j],E->sx[m][3][j],j,E->T[m][j]);
  }
  fflush(E->fp_out);

  return;
}



void read_tic_from_file(struct All_variables *E)
{
  void temperatures_conform_bcs();

  int ii, ll, mm;
  float tt;
  int i, m;
  char output_file[255], input_s[1000];
  FILE *fp;

  float v1, v2, v3, g;

  ii = E->monitor.solution_cycles_init;
  sprintf(output_file,"%s.velo.%d.%d",E->control.old_P_file,E->parallel.me,ii);
  fp=fopen(output_file,"r");
  if (fp == NULL) {
    fprintf(E->fp,"(Initial_temperature.c #1) Cannot open %s\n",output_file);
    parallel_process_termination();
  }

  if (E->parallel.me==0)
    fprintf(E->fp,"Reading %s for initial temperature\n",output_file);

  fgets(input_s,1000,fp);
  sscanf(input_s,"%d %d %f",&ll,&mm,&tt);

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    fgets(input_s,1000,fp);
    sscanf(input_s,"%d %d",&ll,&mm);
    for(i=1;i<=E->lmesh.nno;i++)  {
      fgets(input_s,1000,fp);
      sscanf(input_s,"%g %g %g %f",&(v1),&(v2),&(v3),&(g));

      /* Truncate the temperature to be within (0,1). */
      /* This might not be desirable in some situations. */
      E->T[m][i] = max(0.0,min(g,1.0));
    }
  }
  fclose (fp);

  temperatures_conform_bcs(E);

  return;
}


