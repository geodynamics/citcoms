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
#include <mpi.h>

#include <math.h>
#include <sys/types.h>

#include "element_definitions.h"
#include "global_defs.h"
#include "citcom_init.h"
#include "output.h"
#include "parallel_util.h"
#include "checkpoints.h"

#include "drive_solvers.h"
#include "instructions.h"
#include "interuption.h"
#include "problem_related.h"
#include "process_buoyancy.h"
#include "tracer_setup.h"


void solver_init(struct All_variables *E);


int main(int argc, char **argv)
{	/* Functions called by main*/
  float cpu_time_on_vp_it;

  int cpu_total_seconds,k, *temp;
  double time,initial_time,start_time;

  struct All_variables *E;
  MPI_Comm world;

  MPI_Init(&argc,&argv); /* added here to allow command-line input */

  if (argc < 2)   {
    fprintf(stderr,"Usage: %s PARAMETERFILE\n", argv[0]);
    parallel_process_termination();
  }



  /* this section reads input, allocates memory, and set some initial values;
   *  replaced by CitcomS.Controller.initialize() */
  world = MPI_COMM_WORLD;
  E = citcom_init(&world); /* allocate global E and do initializaion here */

  solver_init(E);

  start_time = time = CPU_time0();
  read_instructions(E, argv[1]);
  initial_setup(E);

  cpu_time_on_vp_it = CPU_time0();
  initial_time = cpu_time_on_vp_it - time;
  if (E->parallel.me == 0)  {
    fprintf(stderr,"Input parameters taken from file '%s'\n",argv[1]);
    fprintf(stderr,"Initialization complete after %g seconds\n\n",initial_time);
    fprintf(E->fp,"Initialization complete after %g seconds\n\n",initial_time);
    fflush(E->fp);
  }



  /* this section sets the initial condition;
   * replaced by CitcomS.Controller.launch() */
  if (E->control.post_p) {
      /* the initial condition is from previous checkpoint */
      read_checkpoint(E);

      /* the program will finish after post_processing */
      post_processing(E);
      (E->problem_output)(E, E->monitor.solution_cycles);
      parallel_process_termination();
  }

  if (E->control.restart) {
      /* the initial condition is from previous checkpoint */
      read_checkpoint(E);
  }
  else {
      /* regular init, or read T from file only */

      initial_conditions(E);

      if(E->control.pseudo_free_surf) {
          if(E->mesh.topvbc == 2)
              general_stokes_solver_pseudo_surf(E);
          else
              assert(0);
      }
      else
          general_stokes_solver(E);
  }

  (E->problem_output)(E, E->monitor.solution_cycles);

  /* information about simulation time and wall clock time */
  output_time(E, E->monitor.solution_cycles);

  if(!E->control.restart)	/* if we have not restarted, print new
				   checkpoint, else leave as is to
				   allow reusing directories */
    output_checkpoint(E);

  /* this section stops the computation if only computes stokes' problem
   * no counterpart in pyre */
  if (E->control.stokes)  {

    if(E->control.tracer==1)
      tracer_advection(E);

    parallel_process_termination();
  }


 

  /* this section advances the time step;
   * replaced by CitcomS.Controller.march() */
  while ( E->control.keep_going   &&  (Emergency_stop == 0) ) {

    /* The next few lines of code were replaced by
     * pyCitcom_PG_timestep_solve() in Pyre version.
     * If you modify here, make sure its Pyre counterpart
     * is modified as well */
    E->monitor.solution_cycles++;
    if(E->monitor.solution_cycles>E->control.print_convergence)
      E->control.print_convergence=1;

    (E->next_buoyancy_field)(E);
    /* */


    if(((E->advection.total_timesteps < E->advection.max_total_timesteps) &&
	(E->advection.timesteps < E->advection.max_timesteps)) ||
       (E->advection.total_timesteps < E->advection.min_timesteps) )
      E->control.keep_going = 1;
    else
      E->control.keep_going = 0;

    cpu_total_seconds = (int)(CPU_time0()-start_time);
    if (cpu_total_seconds > E->control.record_all_until)  {
      E->control.keep_going = 0;
    }

    if (E->monitor.T_interior > E->monitor.T_interior_max_for_exit)  {
      fprintf(E->fp,"quit due to maxT = %.4e sub_iteration%d\n",E->monitor.T_interior,E->advection.last_sub_iterations);
      parallel_process_termination();
    }

    if(E->control.tracer==1)
      tracer_advection(E);

    general_stokes_solver(E);
    if(E->output.write_q_files)
      if ((E->monitor.solution_cycles % E->output.write_q_files)==0)
	heat_flux(E);

    if ((E->monitor.solution_cycles % E->control.record_every)==0) {
	(E->problem_output)(E, E->monitor.solution_cycles);
    }


    /* information about simulation time and wall clock time */
    output_time(E, E->monitor.solution_cycles);

    /* print checkpoint every checkpoint_frequency, unless we have restarted,
       then, we would like to avoid overwriting 
    */
    if ( ((E->monitor.solution_cycles % E->control.checkpoint_frequency)==0) &&
	 ((!E->control.restart) || (E->monitor.solution_cycles != E->monitor.solution_cycles_init))){
	output_checkpoint(E);
    }
    /* updating time-dependent material group
     * if mat_control is 0, the material group has already been
     * initialized in initial_conditions() */
    if(E->control.mat_control==1)
      read_mat_from_file(E);

#ifdef USE_GGRD
    /* updating local rayleigh number (based on Netcdf grds, the
       rayleigh number may be modified laterally in the surface
       layers) */
    /* no counterpart in pyre */
    if(E->control.ggrd.ray_control)
      read_rayleigh_from_file(E);
#endif

    /* updating plate velocity boundary condition */
    if(E->control.vbcs_file==1)
      read_velocity_boundary_from_file(E);

    /* updating plate temperature boundary condition */
    if(E->control.tbcs_file)
      read_temperature_boundary_from_file(E);   


    if (E->parallel.me == 0)  {
      fprintf(E->fp,"CPU total = %g & CPU = %g for step %d time = %.4e dt = %.4e  maxT = %.4e sub_iteration%d\n",CPU_time0()-start_time,CPU_time0()-time,E->monitor.solution_cycles,E->monitor.elapsed_time,E->advection.timestep,E->monitor.T_interior,E->advection.last_sub_iterations);

      time = CPU_time0();
    }

  }



  /* this section prints time accounting;
   * no counterpart in pyre */
  if (E->parallel.me == 0)  {
    fprintf(stderr,"cycles=%d\n",E->monitor.solution_cycles);
    cpu_time_on_vp_it=CPU_time0()-cpu_time_on_vp_it;
    fprintf(stderr,"Average cpu time taken for velocity step = %f\n",
	    cpu_time_on_vp_it/((float)(E->monitor.solution_cycles-E->control.restart)));
    fprintf(E->fp,"Initialization overhead = %f\n",initial_time);
    fprintf(E->fp,"Average cpu time taken for velocity step = %f\n",
	    cpu_time_on_vp_it/((float)(E->monitor.solution_cycles-E->control.restart)));
  }

  output_finalize(E);
  parallel_process_termination();

  return(0);

}
