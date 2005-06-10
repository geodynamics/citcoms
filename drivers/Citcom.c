/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *<LicenseText>
 *=====================================================================
 *
 *                              CitcomS
 *                 ---------------------------------
 *
 *                              Authors:
 *           Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 *           Clint Conrad, Michael Gurnis, and Eun-seo Choi
 *          (c) California Institute of Technology 1994-2005
 *
 *        By downloading and/or installing this software you have
 *       agreed to the CitcomS.py-LICENSE bundled with this software.
 *             Free for non-commercial academic research ONLY.
 *      This program is distributed WITHOUT ANY WARRANTY whatsoever.
 *
 *=====================================================================
 *
 *  Copyright June 2005, by the California Institute of Technology.
 *  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
 * 
 *  Any commercial use must be negotiated with the Office of Technology
 *  Transfer at the California Institute of Technology. This software
 *  may be subject to U.S. export control laws and regulations. By
 *  accepting this software, the user agrees to comply with all
 *  applicable U.S. export laws and regulations, including the
 *  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
 *  the Export Administration Regulations, 15 C.F.R. 730-744. User has
 *  the responsibility to obtain export licenses, or other export
 *  authority as may be required before exporting such information to
 *  foreign countries or providing access to foreign nationals.  In no
 *  event shall the California Institute of Technology be liable to any
 *  party for direct, indirect, special, incidental or consequential
 *  damages, including lost profits, arising out of the use of this
 *  software and its documentation, even if the California Institute of
 *  Technology has been advised of the possibility of such damage.
 * 
 *  The California Institute of Technology specifically disclaims any
 *  warranties, including the implied warranties or merchantability and
 *  fitness for a particular purpose. The software and documentation
 *  provided hereunder is on an "as is" basis, and the California
 *  Institute of Technology has no obligations to provide maintenance,
 *  support, updates, enhancements or modifications.
 *
 *=====================================================================
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
#include "parallel_related.h"

extern int Emergency_stop;

int main(argc,argv)
     int argc;
     char **argv;

{	/* Functions called by main*/
  void general_stokes_solver();
  void general_stokes_solver_pseudo_surf();
  void read_instructions();
  void solve_constrained_flow();
  void solve_derived_velocities();
  void process_temp_field();
  void post_processing();
  void vcopy();
  void construct_mat_group();
  void read_velocity_boundary_from_file();
  void read_mat_from_file();
  void open_time();
  void output();
  void output_pseudo_surf();

  float dot();
  float cpu_time_on_vp_it;

  int cpu_total_seconds,k, *temp;
  double CPU_time0(),time,initial_time,start_time,avaimem();

  struct All_variables *E;
  MPI_Comm world;

  MPI_Init(&argc,&argv); /* added here to allow command-line input */

  if (argc < 2)   {
    fprintf(stderr,"Usage: %s PARAMETERFILE\n", argv[0]);
    parallel_process_termination();
  }

  world = MPI_COMM_WORLD;
  E = citcom_init(&world); /* allocate global E and do initializaion here */

  start_time = time = CPU_time0();
  read_instructions(E, argv[1]);
  open_time(E);

  cpu_time_on_vp_it = CPU_time0();
  initial_time = cpu_time_on_vp_it - time;
  if (E->parallel.me == 0)  {
    fprintf(stderr,"Input parameters taken from file '%s'\n",argv[1]);
    fprintf(stderr,"Initialization complete after %g seconds\n\n",initial_time);
    fprintf(E->fp,"Initialization complete after %g seconds\n\n",initial_time);
    fflush(E->fp);
  }

  if (E->control.post_p)   {
    post_processing(E);
    parallel_process_termination();
  }

  if(E->control.pseudo_free_surf) {
    if(E->mesh.topvbc == 2)
	    general_stokes_solver_pseudo_surf(E);
    else
	    assert(0);
  }
  else
    general_stokes_solver(E);

  if(E->control.pseudo_free_surf) {
    if(E->mesh.topvbc == 2)
	    output_pseudo_surf(E, E->monitor.solution_cycles);
  }
  else
    output(E, E->monitor.solution_cycles);


  if (E->control.stokes)  {

    if(E->control.tracer==1)
      (E->problem_tracer_advection)(E);

    parallel_process_termination();
  }

  while ( E->control.keep_going   &&  (Emergency_stop == 0) )   {
    E->monitor.solution_cycles++;
    if(E->monitor.solution_cycles>E->control.print_convergence)
      E->control.print_convergence=1;

    (E->next_buoyancy_field)(E);

    if(((E->advection.total_timesteps < E->advection.max_total_timesteps) &&
	(E->advection.timesteps < E->advection.max_timesteps)) ||
       (E->advection.total_timesteps < E->advection.min_timesteps) )
      E->control.keep_going = 1;
    else
      E->control.keep_going = 0;

    cpu_total_seconds = CPU_time0()-start_time;
    if (cpu_total_seconds > E->control.record_all_until)  {
      E->control.keep_going = 0;
    }

    if (E->monitor.T_interior>1.5)  {
      fprintf(E->fp,"quit due to maxT = %.4e sub_iteration%d\n",E->monitor.T_interior,E->advection.last_sub_iterations);
      parallel_process_termination();
    }

    general_stokes_solver(E);

    if(E->control.tracer==1)
      (E->problem_tracer_advection)(E);

    if ((E->monitor.solution_cycles % E->control.record_every)==0) {
      if(E->control.pseudo_free_surf) {
        if(E->mesh.topvbc == 2)
	  output_pseudo_surf(E, E->monitor.solution_cycles);
      }
      else
	output(E, E->monitor.solution_cycles);
    }

    if(E->control.mat_control==1)
      read_mat_from_file(E);
    /*
      else
      construct_mat_group(E);
    */

    if(E->control.vbcs_file==1)
      read_velocity_boundary_from_file(E);
    /*
      else
      renew_top_velocity_boundary(E);
    */



    if (E->parallel.me == 0)  {
      fprintf(E->fp,"CPU total = %g & CPU = %g for step %d time = %.4e dt = %.4e  maxT = %.4e sub_iteration%d\n",CPU_time0()-start_time,CPU_time0()-time,E->monitor.solution_cycles,E->monitor.elapsed_time,E->advection.timestep,E->monitor.T_interior,E->advection.last_sub_iterations);
      /* added a time output CPC 6/18/00 */
      fprintf(E->fptime,"%d %.4e %.4e %.4e %.4e\n",E->monitor.solution_cycles,E->monitor.elapsed_time,E->advection.timestep,CPU_time0()-start_time,CPU_time0()-time);
/*       fprintf(stderr,"%d %.4e %.4e %.4e %.4e\n",E->monitor.solution_cycles,E->monitor.elapsed_time,E->advection.timestep,CPU_time0()-start_time,CPU_time0()-time); */
      time = CPU_time0();
    }

  }




  if (E->parallel.me == 0)  {
    fprintf(stderr,"cycles=%d\n",E->monitor.solution_cycles);
    cpu_time_on_vp_it=CPU_time0()-cpu_time_on_vp_it;
    fprintf(stderr,"Average cpu time taken for velocity step = %f\n",
	    cpu_time_on_vp_it/((float)(E->monitor.solution_cycles-E->control.restart)));
    fprintf(E->fp,"Initialization overhead = %f\n",initial_time);
    fprintf(E->fp,"Average cpu time taken for velocity step = %f\n",
	    cpu_time_on_vp_it/((float)(E->monitor.solution_cycles-E->control.restart)));
  }

  fclose(E->fp);
  fclose(E->fptime);

  parallel_process_termination();

  return(0);

}
