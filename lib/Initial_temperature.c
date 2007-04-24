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
void restart_tic_from_file(struct All_variables *);



void tic_input(struct All_variables *E)
{

  int m = E->parallel.me;
  int noz = E->lmesh.noz;
  int n;


  input_int("tic_method", &(E->convection.tic_method), "0,0,2", m);
  /* When tic_method is 0 (default), the temperature is a linear profile +
     perturbation at some layers.

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
  */


  if (E->convection.tic_method == 0 || E->convection.tic_method == 3 ) {
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

  } else if (E->convection.tic_method == 1) {

    input_float("half_space_age", &(E->convection.half_space_age), "40.0,1e-3,nomax", m);

  }
  else if (E->convection.tic_method == 2) {
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
  }
  else {
    fprintf(stderr,"Invalid value of 'tic_method'\n");
    parallel_process_termination();
  }

  return;
}



/* This function is replaced by CitcomS.Components.IC.initTemperature()*/
void convection_initial_temperature(struct All_variables *E)
{
  void report();

  report(E,"Initialize temperature field");
  construct_tic(E);

  /* Note: it is the callee's responsibility to conform tbc. */
  /* like a call to temperatures_conform_bcs(E); */

  if (E->control.verbose)
    debug_tic(E);

  return;
}



void restart_tic(struct All_variables *E)
{
    return;
}


void construct_tic(struct All_variables *E)
{
  if (E->control.lith_age)
    lith_age_construct_tic(E);
  else
    (E->solver.construct_tic_from_input)(E);

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



void restart_tic_from_file(struct All_variables *E)
{
  int ii, ll, mm;
  float restart_elapsed_time;
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
  sscanf(input_s,"%d %d %f",&ll,&mm,&restart_elapsed_time);

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

  return;
}


