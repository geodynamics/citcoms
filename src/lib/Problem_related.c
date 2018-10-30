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
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"

#ifdef USE_GZDIR
int open_file_zipped(char *, FILE **,struct All_variables *);
void gzip_file(char *);
#endif
/*=======================================================================
  read velocity vectors at the top surface from files
=========================================================================*/

void read_velocity_boundary_from_file(E)
     struct All_variables *E;
{
    (E->solver.read_input_files_for_timesteps)(E,1,1); /* read velocity(1) and output(1) */
    return;
}
#ifdef USE_GGRD
/* 

wrapper for ggrd functionality to read in netcdf grid files for
laterally varying rayleigh number in the top layers 

 */
void read_rayleigh_from_file(E)
     struct All_variables *E;
{
  (E->solver.read_input_files_for_timesteps)(E,4,1); /* read Rayleigh number for top layers */
  return;
}
#endif
/*=======================================================================
  construct material array
=========================================================================*/

void read_mat_from_file(E)
     struct All_variables *E;
{
    (E->solver.read_input_files_for_timesteps)(E,3,1); /* read element material(3) and output(1) */
  return;

}
/*=======================================================================
  read temperature at the top surface from files
=========================================================================*/

void read_temperature_boundary_from_file(E)
     struct All_variables *E;
{
    (E->solver.read_input_files_for_timesteps)(E,5,1); /* read temperature(5) and output(1) */
    return;
}

/*=======================================================================
  read temperature and stencils for slab assimilation from files
=========================================================================*/
/* DJB SLAB */
void read_slab_temperature_from_file(E)
     struct All_variables *E;
{
    /* debugging */
    /*if( E->parallel.me == 0)
        fprintf(stderr, "read_slab_temperature_from_file, just before call with 6,1\n");*/

    (E->solver.read_input_files_for_timesteps)(E,6,1); /* read temperature and stencil (6) and output(1) */
    return;
}

/*=======================================================================
  read internal (slab) velocity from files
=========================================================================*/
/* DJB SLAB */

void read_internal_velocity_from_file(E)
     struct All_variables *E; 
{
    /* debugging */
    /*if( E->parallel.me == 0)
        fprintf(stderr, "read_internal_velocity_from_file, just before call with 7,1\n"); */

    (E->solver.read_input_files_for_timesteps)(E,7,1); /* read velocity and stencil (7) and output(1) */
    return;
}

/*=======================================================================
  Open restart file to get initial elapsed time, or calculate the right value
=========================================================================*/

void get_initial_elapsed_time(E)
  struct All_variables *E;
{
    FILE *fp;
    int ll, mm,rezip;
    char output_file[255],input_s[1000];

    E->monitor.elapsed_time = 0.0;

    if (E->convection.tic_method == -1) {

#ifdef USE_GZDIR		/* gzdir output */
      if(strcmp(E->output.format, "ascii-gz") == 0){
	if(E->output.gzdir.vtk_io)
	  sprintf(output_file, "%s/%d/t.%d.%d",
		  E->control.data_dir_old,E->monitor.solution_cycles_init,E->parallel.me,E->monitor.solution_cycles_init);
	else
	  sprintf(output_file, "%s/%d/velo.%d.%d",
		  E->control.data_dir_old,E->monitor.solution_cycles_init,E->parallel.me,E->monitor.solution_cycles_init);
      }else{
	sprintf(output_file, "%s.velo.%d.%d",E->control.old_P_file,E->parallel.me,E->monitor.solution_cycles_init);
      }
      rezip = open_file_zipped(output_file,&fp,E);
#else  /* all others */
      sprintf(output_file, "%s.velo.%d.%d",E->control.old_P_file,E->parallel.me,E->monitor.solution_cycles_init);
      fp=fopen(output_file,"r");
#endif

      if (fp == NULL) {
	fprintf(E->fp,"(Problem_related #8) Cannot open %s\n",output_file);
	exit(8);
      }
      fgets(input_s,1000,fp);
      if(sscanf(input_s,"%d %d %f",&ll,&mm,&E->monitor.elapsed_time) != 3) {
        fprintf(stderr,"Error while reading file '%s'\n", output_file);
        exit(8);
      }
      fclose(fp);
#ifdef USE_GZDIR
      if(rezip)
	gzip_file(output_file);
#endif
    } /* end if tic_method */

    return;
}

/*=======================================================================
  Sets the elapsed time to zero, if desired.
=========================================================================*/

void set_elapsed_time(E)
  struct All_variables *E;
{

    if (E->control.zero_elapsed_time) /* set elapsed_time to zero */
	E->monitor.elapsed_time = 0.0;

   return;
}

/*=======================================================================
  Resets the age at which to start time (startage) to the end of the previous
  run, if desired.
=========================================================================*/

void set_starting_age(E)
  struct All_variables *E;
{
/* remember start_age is in MY */
    if (E->control.reset_startage)
	E->control.start_age = E->monitor.elapsed_time*E->data.scalet;

   return;
}


/*=======================================================================
  Returns age at which to open an input file (velocity, material, age)
  NOTE: Remember that ages are positive, but going forward in time means
  making ages SMALLER!
=========================================================================*/

  float find_age_in_MY(E)

  struct All_variables *E;
{
   float age_in_MY, e_4;


   e_4=1.e-4;

   if (E->data.timedir >= 0) { /* forward convection */
      age_in_MY = E->control.start_age - E->monitor.elapsed_time*E->data.scalet;
   }
   else { /* backward convection */
      age_in_MY = E->control.start_age + E->monitor.elapsed_time*E->data.scalet;
   }

      if (((age_in_MY+e_4) < 0.0) && (E->monitor.solution_cycles < 1)) {
        if (E->parallel.me == 0) fprintf(stderr,"Age = %g Ma, Initial age should not be negative!\n",age_in_MY);
	exit(11);
      }

   return(age_in_MY);
}
