#include <mpi.h>

#include <math.h>
#include <sys/types.h>

#include "element_definitions.h"
#include "global_defs.h"
#include "citcom_init.h"

extern int Emergency_stop;

int main(argc,argv)
     int argc;
     char **argv;
     
{	/* Functions called by main*/
  void general_stokes_solver();
  void read_instructions();
  void solve_constrained_flow();
  void solve_derived_velocities();
  void parallel_process_initilization();
  void parallel_process_termination();
  void parallel_process_sync();
  void process_temp_field(); 
  void process_output_field(); 
  void vcopy();
  void construct_mat_group();
  void read_velocity_boundary_from_file();
  void read_mat_from_file();

  float dot();

  int cpu_total_seconds,k, *temp;
  double CPU_time0(),time,initial_time,start_time,avaimem();
    
  MPI_Init(&argc,&argv); /* added here to allow command-line input */
  E = Citcom_Init();
  
  parallel_process_initilization(E,argc,argv);
  
  E->monitor.solution_cycles=0;
  
  start_time = time = CPU_time0();
  read_instructions(argc,argv);

  E->control.keep_going=1;

  if (E->parallel.me == 0)  {
    fprintf(stderr,"Input parameters taken from file '%s'\n",argv[1]);
    fprintf(stderr,"Initialization complete after %g seconds\n\n",CPU_time0()-time); fflush(E->fp);
    initial_time = CPU_time0()-time;
    E->monitor.cpu_time_on_vp_it = CPU_time0();
  }
  
  if (E->control.post_p)   {
    process_output_field(E,E->monitor.solution_cycles);
    parallel_process_termination();
  }

  general_stokes_solver(E);

  /* moved up here so that the first time step would be outputted CPC 3/21/00 */
  process_temp_field(E,E->monitor.solution_cycles);
  process_new_velocity(E,E->monitor.solution_cycles);

  if (E->control.stokes)  {
    
    /*     process_temp_field(E,E->monitor.solution_cycles);
	   process_new_velocity(E,E->monitor.solution_cycles); */
    
    /*      if(E->control.tracer==1)  {
	    (E->problem_tracer_advection)(E);
	    (E->problem_tracer_output)(E,E->monitor.solution_cycles);
	    }
    */    
    parallel_process_termination();
  }
  
  E->control.DIRECTII = 0;
  
  while ( E->control.keep_going   &&  (Emergency_stop == 0) )   {
    
    
    E->monitor.solution_cycles++; 
    if(E->monitor.solution_cycles>E->control.print_convergence)
      E->control.print_convergence=1;
    
    
    (E->next_buoyancy_field)(E);
    
    cpu_total_seconds = CPU_time0()-start_time;
    if (cpu_total_seconds > E->control.record_all_until)  {
      E->control.DIRECTII = 1;
      E->control.keep_going = 0;
    }
    
    process_temp_field(E,E->monitor.solution_cycles); 
    
    if (E->monitor.T_interior>1.5)  {
      fprintf(E->fp,"quit due to maxT = %.4e sub_iteration%d\n",E->monitor.T_interior,E->advection.last_sub_iterations);
      parallel_process_termination();
    }
    
    general_stokes_solver(E);
    process_new_velocity(E,E->monitor.solution_cycles);
    
    /*      if(E->control.tracer==1)  {
	    (E->problem_tracer_advection)(E);
	    (E->problem_tracer_output)(E,E->monitor.solution_cycles);
	    }
    */
    
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
      fprintf(stderr,"%d %.4e %.4e %.4e %.4e\n",E->monitor.solution_cycles,E->monitor.elapsed_time,E->advection.timestep,CPU_time0()-start_time,CPU_time0()-time);
      time = CPU_time0();
    }
    
  }
  
  
  
  
  if (E->parallel.me == 0)  {
    fprintf(stderr,"cycles=%d\n",E->monitor.solution_cycles);
    E->monitor.cpu_time_on_vp_it=CPU_time0()-E->monitor.cpu_time_on_vp_it;
    fprintf(stderr,"Average cpu time taken for velocity step = %f\n",
	    E->monitor.cpu_time_on_vp_it/((float)(E->monitor.solution_cycles-E->control.restart)));
    fprintf(E->fp,"Initialization overhead = %f\n",initial_time);
    fprintf(E->fp,"Average cpu time taken for velocity step = %f\n",
	    E->monitor.cpu_time_on_vp_it/((float)(E->monitor.solution_cycles-E->control.restart)));
  }
  
  fclose(E->fp);
  fclose(E->fptime);
  
  parallel_process_termination();
  
  return(0);  
  
} 
