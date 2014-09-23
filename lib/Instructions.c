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
/* Set up the finite element problem to suit: returns with all memory */
/* allocated, temperature, viscosity, node locations and how to use */
/* them all established. 8.29.92 or 29.8.92 depending on your nationality*/

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <sys/stat.h>
#include <sys/errno.h>
#include <unistd.h>
#include <ctype.h>
#include "element_definitions.h"
#include "global_defs.h"

#include "citcom_init.h"
#include "initial_temperature.h"
#include "lith_age.h"
#include "material_properties.h"
#include "output.h"
#include "output_h5.h"
#include "parallel_related.h"
#include "parsing.h"
#include "phase_change.h"

void parallel_process_termination();
void allocate_common_vars(struct All_variables*);
void allocate_velocity_vars(struct All_variables*);
void check_bc_consistency(struct All_variables*);
void construct_elt_gs(struct All_variables*);
void construct_elt_cs(struct All_variables*);
void construct_shape_function_derivatives(struct All_variables *E);
void construct_id(struct All_variables*);
void construct_ien(struct All_variables*);
void construct_lm(struct All_variables*);
void construct_masks(struct All_variables*);
void construct_shape_functions(struct All_variables*);
void construct_sub_element(struct All_variables*);
void construct_surf_det (struct All_variables*);
void construct_bdry_det (struct All_variables*);
void construct_surface (struct All_variables*);
void get_initial_elapsed_time(struct All_variables*);
void lith_age_init(struct All_variables *E);
void mass_matrix(struct All_variables*);
void output_init(struct All_variables*);
void set_elapsed_time(struct All_variables*);
void set_sphere_harmonics (struct All_variables*);
void set_starting_age(struct All_variables*);
void tracer_initial_settings(struct All_variables*);
void tracer_input(struct All_variables*);
void viscosity_input(struct All_variables*);
void vtk_output(struct All_variables*, int);
void get_vtk_filename(char *,int,struct All_variables *,int);
void myerror(struct All_variables *,char *);
void open_qfiles(struct All_variables *) ;
void read_rayleigh_from_file(struct All_variables *);
void read_initial_settings(struct All_variables *);
void check_settings_consistency(struct All_variables *);
void global_derived_values(struct All_variables *);


void initial_mesh_solver_setup(struct All_variables *E)
{
  int chatty;
  //chatty = ((E->parallel.me == 0)&&(E->control.verbose))?(1):(0);
  chatty = E->parallel.me == 0;

    E->monitor.cpu_time_at_last_cycle =
        E->monitor.cpu_time_at_start = CPU_time0();

    output_init(E);
    (E->problem_derived_values)(E);   /* call this before global_derived_  */
    global_derived_values(E);

    (E->solver.parallel_processor_setup)(E);   /* get # of proc in x,y,z */
    (E->solver.parallel_domain_decomp0)(E);  /* get local nel, nno, elx, nox et al */

    allocate_common_vars(E);
    (E->problem_allocate_vars)(E);
    (E->solver_allocate_vars)(E);
    if(chatty)fprintf(stderr,"memory allocation done\n");
           /* logical domain */
    construct_ien(E);
    construct_surface(E);
    (E->solver.construct_boundary)(E);
    (E->solver.parallel_domain_boundary_nodes)(E);
    if(chatty)fprintf(stderr,"parallel setup done\n");

           /* physical domain */
    (E->solver.node_locations)(E);
    if(chatty)fprintf(stderr,"node locations done\n");

    allocate_velocity_vars(E);
    if(chatty)fprintf(stderr,"velocity vars done\n");


    get_initial_elapsed_time(E);  /* Set elapsed time */
    set_starting_age(E);  /* set the starting age to elapsed time, if desired */
    set_elapsed_time(E);         /* reset to elapsed time to zero, if desired */


    /* open the heatflow files here because we need to know about loc_me */
    if(E->output.write_q_files)
      open_qfiles(E);
    else{
      E->output.fpqt = E->output.fpqb = NULL;
    }



    if(E->control.lith_age)
        lith_age_init(E);

    (E->problem_boundary_conds)(E);

    check_bc_consistency(E);
    if(chatty)fprintf(stderr,"boundary conditions done\n");

    construct_masks(E);		/* order is important here */
    construct_id(E);
    construct_lm(E);
    if(chatty)fprintf(stderr,"id/lm done\n");

    (E->solver.parallel_communication_routs_v)(E);
    if(chatty)fprintf(stderr,"v communications done\n");

    if(E->control.use_cbf_topo){
      (E->solver.parallel_communication_routs_s)(E); 
      if(chatty)fprintf(stderr,"s communications done\n");
    }
    reference_state(E);

    construct_sub_element(E);
    construct_shape_functions(E);
    construct_shape_function_derivatives(E);
    if(chatty)fprintf(stderr,"shape functions done\n");

    construct_elt_gs(E);


    if(E->control.inv_gruneisen != 0)
        construct_elt_cs(E);

    /* this matrix results from spherical geometry */
    /* construct_c3x3matrix(E); */

    mass_matrix(E);

    construct_surf_det (E);
    construct_bdry_det (E);

    if(chatty)fprintf(stderr,"mass matrix, dets done\n");

    set_sphere_harmonics (E);


    if(E->control.tracer) {
	tracer_initial_settings(E);
	(E->problem_tracer_setup)(E);
	if(chatty)fprintf(stderr,"tracer setup done\n");
    }

#ifdef USE_GGRD
    /* updating local rayleigh number (based on Netcdf grds, the
       rayleigh number may be modified laterally in the surface
       layers) */
    if(E->control.ggrd.ray_control)
      read_rayleigh_from_file(E);
#endif

    if(chatty)fprintf(stderr,"initial_mesh_solver_setup done\n");
}


/* This function is replaced by CitcomS.*.setProperties() in Pyre. */
void read_instructions(struct All_variables *E, char *filename)
{
    void read_initial_settings();

    void setup_parser();
    void shutdown_parser();

    /* ==================================================
       Initialize from the command line
       from startup files. (See Parsing.c).
       ==================================================  */

    setup_parser(E,filename);
    read_initial_settings(E);
    shutdown_parser(E);

    return;
}


/* This function is replaced by CitcomS.Solver.initial_setup() in Pyre. */
void initial_setup(struct All_variables *E)
{
    initial_mesh_solver_setup(E);
    general_stokes_solver_setup(E);
    (E->next_buoyancy_field_init)(E);
}

void initialize_material(struct All_variables *E)
{
    void construct_mat_group();
    void read_mat_from_file();

    if(E->control.mat_control)
        read_mat_from_file(E);
    else
        construct_mat_group(E);
}


/* This function is replaced by CitcomS.Components.IC.launch()*/
void initial_conditions(struct All_variables *E)
{
    void initialize_tracers();
    void init_composition();
    void common_initial_fields();

    initialize_material(E);

    if (E->control.tracer==1) {
        initialize_tracers(E);

        if (E->composition.on)
            init_composition(E);
    }

    (E->problem_initial_fields)(E);   /* temperature/chemistry/melting etc */
    common_initial_fields(E);  /* velocity/pressure/viscosity (viscosity must be done LAST) */

    return;
}


void read_initial_settings(struct All_variables *E)
{
  void set_convection_defaults();
  void set_cg_defaults();
  void set_mg_defaults();
  float tmp;
  double ell_tmp;
  int m=E->parallel.me,i;
  double levmax;
  int tmp_int_in;

  /* first the problem type (defines subsequent behaviour) */

  input_string("Problem",E->control.PROBLEM_TYPE,"convection",m);

  if ( strcmp(E->control.PROBLEM_TYPE,"convection") == 0)
    set_convection_defaults(E);
  else if ( strcmp(E->control.PROBLEM_TYPE,"convection-chemical") == 0)
    set_convection_defaults(E);
  else {
    fprintf(E->fp,"Unable to determine problem type, assuming convection ... \n");
    set_convection_defaults(E);
  }

  input_string("Geometry",E->control.GEOMETRY,"sphere",m);
  if ( strcmp(E->control.GEOMETRY,"sphere") == 0)
      (E->solver.set_3dsphere_defaults)(E);
  else {
    fprintf(E->fp,"Unable to determine geometry, assuming sphere 3d ... \n");
    (E->solver.set_3dsphere_defaults)(E);
  }

  input_string("Solver",E->control.SOLVER_TYPE,"cgrad",m);
  if ( strcmp(E->control.SOLVER_TYPE,"cgrad") == 0)
    set_cg_defaults(E);
  else if ( strcmp(E->control.SOLVER_TYPE,"multigrid") == 0)
    set_mg_defaults(E);
  else {
    if (E->parallel.me==0) fprintf(stderr,"Unable to determine how to solve, specify Solver=VALID_OPTION \n");
    parallel_process_termination();
  }


  /* Information on which files to print, which variables of the flow to calculate and print.
     Default is no information recorded (apart from special things for given applications.
  */

  input_string("datadir",E->control.data_dir,".",m);
  input_string("datafile",E->control.data_prefix,"initialize",m);
  input_string("datadir_old",E->control.data_dir_old,".",m);
  input_string("datafile_old",E->control.data_prefix_old,"initialize",m);

  input_int("nproc_surf",&(E->parallel.nprocxy),"1",m);
  input_int("nprocx",&(E->parallel.nprocx),"1",m);
  input_int("nprocy",&(E->parallel.nprocy),"1",m);
  input_int("nprocz",&(E->parallel.nprocz),"1",m);

  if (E->control.CONJ_GRAD) {
      input_int("nodex",&(E->mesh.nox),"essential",m);
      input_int("nodez",&(E->mesh.noz),"essential",m);
      input_int("nodey",&(E->mesh.noy),"essential",m);

      E->mesh.mgunitx = (E->mesh.nox - 1) / E->parallel.nprocx;
      E->mesh.mgunity = (E->mesh.noy - 1) / E->parallel.nprocy;
      E->mesh.mgunitz = (E->mesh.noz - 1) / E->parallel.nprocz;
      E->mesh.levels = 1;
  }
  else {
      input_int("mgunitx",&(E->mesh.mgunitx),"1",m);
      input_int("mgunitz",&(E->mesh.mgunitz),"1",m);
      input_int("mgunity",&(E->mesh.mgunity),"1",m);

      input_int("levels",&(E->mesh.levels),"1",m);

      levmax = E->mesh.levels - 1;
      E->mesh.nox = E->mesh.mgunitx * (int) pow(2.0,levmax) * E->parallel.nprocx + 1;
      E->mesh.noy = E->mesh.mgunity * (int) pow(2.0,levmax) * E->parallel.nprocy + 1;
      E->mesh.noz = E->mesh.mgunitz * (int) pow(2.0,levmax) * E->parallel.nprocz + 1;
  }

  input_double("radius_outer",&(E->sphere.ro),"1",m);
  input_double("radius_inner",&(E->sphere.ri),"0.55",m);

  if(E->sphere.caps == 1) {
      input_double("theta_min",&(E->control.theta_min),"essential",m);
      input_double("theta_max",&(E->control.theta_max),"essential",m);
      input_double("fi_min",&(E->control.fi_min),"essential",m);
      input_double("fi_max",&(E->control.fi_max),"essential",m);
  }

  input_int("coor",&(E->control.coor),"0",m);
  if(E->control.coor == 2){
    /*
       refinement in two layers
    */
    /* number of refinement layers */
    E->control.coor_refine[0] = 0.10; /* bottom 10% */
    E->control.coor_refine[1] = 0.15; /* get 15% of the nodes */
    E->control.coor_refine[2] = 0.10; /* top 10% */
    E->control.coor_refine[3] = 0.20; /* get 20% of the nodes */
    input_float_vector("coor_refine",4,E->control.coor_refine,m);
  }else if(E->control.coor == 3){
    /*

    refinement CitcomCU style, by reading in layers, e.g.

	r_grid_layers=3		# minus 1 is number of layers with uniform grid in r
	rr=0.5,0.75,1.0 	#    starting and ending r coodinates
	nr=1,37,97		#    starting and ending node in r direction

    */
    input_int("r_grid_layers", &(E->control.rlayers), "1",m);
    if(E->control.rlayers > 20)
      myerror(E,"number of rlayers out of bounds (20) for coor = 3");
    /* layers radii */
    input_float_vector("rr", E->control.rlayers, (E->control.rrlayer),m);
    /* associated node numbers */
    input_int_vector("nr", E->control.rlayers, (E->control.nrlayer),m);
   }

  input_string("coor_file",E->control.coor_file,"",m);


  input_boolean("node_assemble",&(E->control.NASSEMBLE),"off",m);
  /* general mesh structure */

  input_boolean("verbose",&(E->control.verbose),"off",m);
  input_boolean("see_convergence",&(E->control.print_convergence),"off",m);

  input_boolean("stokes_flow_only",&(E->control.stokes),"off",m);

  //input_boolean("remove_hor_buoy_avg",&(E->control.remove_hor_buoy_avg),"on",m);


  /* restart from checkpoint file */
  input_boolean("restart",&(E->control.restart),"off",m);
  input_int("post_p",&(E->control.post_p),"0",m);
  input_int("solution_cycles_init",&(E->monitor.solution_cycles_init),"0",m);

  /* for layers    */

  input_int("num_mat",&(E->viscosity.num_mat),"1",m); /* number of layers, moved
							 from Viscosity_structures.c */
  if(E->viscosity.num_mat > CITCOM_MAX_VISC_LAYER)
    myerror(E,"too many viscosity layers as per num_mat, increase CITCOM_MAX_VISC_LAYER");

  /* those are specific depth layers associated with phase
     transitions, default values should be fixed */
  input_float("z_cmb",&(E->viscosity.zcmb),"0.45",m); /* 0.45063569 */
  input_float("z_lmantle",&(E->viscosity.zlm),"0.103594412180191",m); /*0.10359441  */
  input_float("z_410",&(E->viscosity.z410),"0.0643541045361796",m); /* 0.06434, more like it */
  input_float("z_lith",&(E->viscosity.zlith),"0.0156961230576048",m); /* 0.0157, more like it */


  /* those are depth layers associated with viscosity or material
     jumps, they may or may not be identical with the phase changes */
  E->viscosity.zbase_layer[0] = E->viscosity.zbase_layer[1] = -999;
  input_float_vector("z_layer",E->viscosity.num_mat,(E->viscosity.zbase_layer),m);



  /*  the start age and initial subduction history   */
  input_float("start_age",&(E->control.start_age),"0.0",m);
  input_int("reset_startage",&(E->control.reset_startage),"0",m);
  input_int("zero_elapsed_time",&(E->control.zero_elapsed_time),"0",m);

  input_int("output_ll_max",&(E->output.llmax),"1",m);

  input_int("topvbc",&(E->mesh.topvbc),"0",m);
  input_int("botvbc",&(E->mesh.botvbc),"0",m);


  /* 

  internal boundary conditions

  */
  input_int("toplayerbc",&(E->mesh.toplayerbc),"0",m); /* > 0: apply surface boundary condition
                                                            throughout all nodes with r > toplayerbc_r
							    < 0: apply to single node layer noz+toplayerbc

						       */
  input_float("toplayerbc_r",&(E->mesh.toplayerbc_r),"0.984303876942",m); /* minimum r to apply BC to, 
									     100 km depth */




  input_float("topvbxval",&(E->control.VBXtopval),"0.0",m);
  input_float("botvbxval",&(E->control.VBXbotval),"0.0",m);
  input_float("topvbyval",&(E->control.VBYtopval),"0.0",m);
  input_float("botvbyval",&(E->control.VBYbotval),"0.0",m);


  input_float("T_interior_max_for_exit",&(E->monitor.T_interior_max_for_exit),"1.5",m);

  input_int("pseudo_free_surf",&(E->control.pseudo_free_surf),"0",m);

  input_int("toptbc",&(E->mesh.toptbc),"1",m);
  input_int("bottbc",&(E->mesh.bottbc),"1",m);
  input_float("toptbcval",&(E->control.TBCtopval),"0.0",m);
  input_float("bottbcval",&(E->control.TBCbotval),"1.0",m);

  input_boolean("side_sbcs",&(E->control.side_sbcs),"off",m);

  input_int("file_vbcs",&(E->control.vbcs_file),"0",m);
  input_string("vel_bound_file",E->control.velocity_boundary_file,"",m);

  input_int("file_tbcs",&(E->control.tbcs_file),"0",m);
  input_string("temp_bound_file",E->control.temperature_boundary_file,"",m);

  input_int("reference_state",&(E->refstate.choice),"1",m);
  if(E->refstate.choice == 0) {
      input_string("refstate_file",E->refstate.filename,"refstate.dat",m);
  }

  input_int("mineral_physics_model",&(E->control.mineral_physics_model),"1",m);

  input_int("mat_control",&(E->control.mat_control),"0",m);
  input_string("mat_file",E->control.mat_file,"",m);


  input_boolean("precise_strain_rate",&(E->control.precise_strain_rate),"off",m);

#ifdef USE_GGRD


  /* 
     
  note that this part of the code might override mat_control, file_vbcs,
     
  MATERIAL CONTROL 

     usage:
     (a) 

     ggrd_mat_control=2
     ggrd_mat_file="weak.grd"

     read in time-constant prefactors from weak.grd netcdf file that apply to top two E->mat layers

     i.e.  ggrd_mat_control > 0 --> assign to layers with ilayer <=   ggrd_mat_control
           ggrd_mat_control < 0 --> assign to layers with ilayer ==  -ggrd_mat_control


     (b)

     ggrd_mat_control=2
     ggrd_mat_file="mythist"
     ggrd_time_hist_file="mythist/times.dat"


     time-dependent, will look for n files named mythist/i/weak.grd
     where i = 1...n and n is the number of times as specified in
     ggrd_time_hist_file which has time in Ma for n stages like so

     -->age is positive, and forward marching in time decreases the age<--
     
     0 15
     15 30
     30 60



     in the example above, the input grid is a layer. if it's a 3D model, provide 
     ggrd_mat_depth_file, akin to temperature input
     
     
  */
  ggrd_init_master(&E->control.ggrd);
  /* this is controlling velocities, material, and age */
  /* time history file, if not specified, will use constant VBCs and material grids */
  input_string("ggrd_time_hist_file",
	       E->control.ggrd.time_hist.file,"",m); 
  /* if > 0, will use top  E->control.ggrd.mat_control layers and assign a prefactor for the viscosity */
  /* if < 0, will assign only to layer == -ggrd_mat_control */
  input_int("ggrd_mat_control",&(E->control.ggrd.mat_control),"0",m); 
  input_boolean("ggrd_mat_limit_prefactor",&(E->control.ggrd_mat_limit_prefactor),"on",m); /* limit prefactor to with 1e+/-5 */
  input_int("ggrd_mat_is_code",&(E->control.ggrd_mat_is_code),"0",m); /* the viscosity grids are
									   actually codes for
									   different types of
									   rheologies, from 1 .... cmax
									   
									   
									*/
  if(E->control.ggrd_mat_is_code){
    /* we need code assignments */
    E->control.ggrd_mat_code_viscosities = (float *)malloc(sizeof(float)*E->control.ggrd_mat_is_code);
    for(i=0;i < E->control.ggrd_mat_is_code;i++)
      E->control.ggrd_mat_code_viscosities[i] = 1;
    input_float_vector("ggrd_mat_code_viscosities",
		       E->control.ggrd_mat_is_code,(E->control.ggrd_mat_code_viscosities),m);
  }
  input_string("ggrd_mat_file",E->control.ggrd.mat_file,"",m); /* file to read prefactors from */
  input_string("ggrd_mat_depth_file",
	       E->control.ggrd_mat_depth_file,"_i_do_not_exist_",m); 
  if(E->control.ggrd.mat_control != 0) /* this will override mat_control setting */
    E->control.mat_control = 1;
  /* 
     
  Surface layer Rayleigh number control, similar to above

  */
  input_int("ggrd_rayleigh_control",
	    &(E->control.ggrd.ray_control),"0",m); 
  input_string("ggrd_rayleigh_file",
	       E->control.ggrd.ray_file,"",m); /* file to read prefactors from */
  /* 
     
  surface velocity control, similar to material control above
  
  if time-dependent, will look for ggrd_vtop_file/i/v?.grd
  if constant, will look for ggrd_vtop_file/v?.grd

  where vp/vt.grd are Netcdf GRD files with East and South velocities in cm/yr
  

  */
  input_int("ggrd_vtop_control",&(E->control.ggrd.vtop_control),"0",m); 
  input_string("ggrd_vtop_dir",E->control.ggrd.vtop_dir,"",m); /* file to read prefactors from */

  /* 
     
  if ggrd_vtop_euler is set, will read 

  wx wy wz 

  from E->control.ggrd.vtop_dir/rotvec.dat

  and location codes from E->control.ggrd.vtop_dir/code.grd

  assigning euler vector velocities in cm/yr assuming wx/wy/wz are in deg/Myr

  codes go between 1....N where N is the number of entries in rotvec.dat

  */
  input_boolean("ggrd_vtop_euler",&(E->control.ggrd_vtop_euler),"off",m);
  if(E->control.ggrd_vtop_euler)
    E->control.ggrd.vtop_control = 1;

  if(E->control.ggrd.vtop_control) /* this will override mat_control setting */
    E->control.vbcs_file = 1;

  /* if set, will check the theta velocities from grid input for
     scaled (internal non dim) values of > 1e9. if found, those nodes will
     be set to free slip
  */
  input_boolean("allow_mixed_vbcs",&(E->control.ggrd_allow_mixed_vbcs),"off",m);

  /* when assigning  composition from grid file, allow values between 0 and 1 ? 
     default will ronud up/down to 0 or 1
  */
  input_boolean("ggrd_comp_smooth",&(E->control.ggrd_comp_smooth),"off",m);


#endif

  input_boolean("aug_lagr",&(E->control.augmented_Lagr),"off",m);
  input_double("aug_number",&(E->control.augmented),"0.0",m);

  input_boolean("remove_rigid_rotation",&(E->control.remove_rigid_rotation),"on",m);
  input_boolean("inner_remove_rigid_rotation",&(E->control.inner_remove_rigid_rotation),"off",m);
  input_boolean("remove_angular_momentum",&(E->control.remove_angular_momentum),"on",m);

  input_boolean("self_gravitation",&(E->control.self_gravitation),"off",m);
  input_boolean("use_cbf_topo",&(E->control.use_cbf_topo),"off",m); /* make default on later XXX TWB */


  input_int("storage_spacing",&(E->control.record_every),"10",m);
  input_int("checkpointFrequency",&(E->control.checkpoint_frequency),"100",m);
  input_int("cpu_limits_in_seconds",&(E->control.record_all_until),"5",m);
  input_int("write_q_files",&(E->output.write_q_files),"0",m);/* write additional
								 heat flux files? */
  if(E->output.write_q_files){	/* make sure those get written at
				   least as often as velocities */
    E->output.write_q_files = min(E->output.write_q_files,E->control.record_every);
  }


  input_boolean("precond",&(E->control.precondition),"off",m);

  input_int("mg_cycle",&(E->control.mg_cycle),"2,0,nomax",m);
  input_int("down_heavy",&(E->control.down_heavy),"1,0,nomax",m);
  input_int("up_heavy",&(E->control.up_heavy),"1,0,nomax",m);
  input_double("accuracy",&(E->control.accuracy),"1.0e-4,0.0,1.0",m);
  input_double("inner_accuracy_scale",&(E->control.inner_accuracy_scale),"1.0,0.000001,1.0",m);

  input_boolean("force_iteration",&(E->control.force_iteration),"off",m);

  input_boolean("check_continuity_convergence",&(E->control.check_continuity_convergence),"on",m);
  input_boolean("check_pressure_convergence",&(E->control.check_pressure_convergence),"on",m);

  /* for backward compatibility, override */
  input_boolean("only_check_vel_convergence",&tmp_int_in,"off",m);
  if(tmp_int_in){
    E->control.check_continuity_convergence = 0;
    E->control.check_pressure_convergence = 0;
  }

  input_int("vhighstep",&(E->control.v_steps_high),"1,0,nomax",m);
  input_int("vlowstep",&(E->control.v_steps_low),"250,0,nomax",m);
  input_int("max_mg_cycles",&(E->control.max_mg_cycles),"50,0,nomax",m);
  input_int("piterations",&(E->control.p_iterations),"100,0,nomax",m);

  input_float("rayleigh",&(E->control.Atemp),"essential",m);

  input_float("dissipation_number",&(E->control.disptn_number),"0.0",m);
  input_float("gruneisen",&(tmp),"0.0",m);
  /* special case: if tmp==0, set gruneisen as inf */
  if(tmp != 0)
      E->control.inv_gruneisen = 1/tmp;
  else
      E->control.inv_gruneisen = 0;

  if(E->control.inv_gruneisen != 0) {
      /* which compressible solver to use: "cg" or "bicg" */
      input_string("uzawa",E->control.uzawa,"cg",m);
      if(strcmp(E->control.uzawa, "cg") == 0) {
          /* more convergence parameters for "cg" */
          input_int("compress_iter_maxstep",&(E->control.compress_iter_maxstep),"100",m);
      }
      else if(strcmp(E->control.uzawa, "bicg") == 0) {
      }
      else
          myerror(E, "Error: unknown Uzawa iteration\n");
  }

  input_float("surfaceT",&(E->control.surface_temp),"0.1",m);
  /*input_float("adiabaticT0",&(E->control.adiabaticT0),"0.4",m);*/
  input_float("Q0",&(E->control.Q0),"0.0",m);
  /* Q0_enriched gets read in Tracer_setup.c */

  /* data section */
  input_float("gravacc",&(E->data.grav_acc),"9.81",m);
  input_float("thermexp",&(E->data.therm_exp),"3.0e-5",m);
  input_float("cp",&(E->data.Cp),"1200.0",m);
  input_float("thermdiff",&(E->data.therm_diff),"1.0e-6",m);
  input_float("density",&(E->data.density),"3340.0",m);
  input_float("density_above",&(E->data.density_above),"1030.0",m);
  input_float("density_below",&(E->data.density_below),"6600.0",m);
  input_float("refvisc",&(E->data.ref_viscosity),"1.0e21",m);


  input_double("ellipticity",&ell_tmp,"0.0",m);
#ifdef ALLOW_ELLIPTICAL
  /* 

  ellipticity and rotation settings
  
  */
  /* f = (a-c)/a, where c is the short, a=b the long axis 
     1/298.257 = 0.00335281317789691 for Earth at present day
  */
  E->data.ellipticity = ell_tmp;
  if(fabs(E->data.ellipticity) > 5e-7){

    /* define ra and rc such that R=1 is the volume equivalanet */
    E->data.ra = pow((1.-E->data.ellipticity),-1./3.); /* non dim long axis */
    E->data.rc = 1./(E->data.ra * E->data.ra); /* non dim short axis */
    E->data.efac = (1.-E->data.ellipticity)*(1.-E->data.ellipticity);
    if(E->parallel.me == 0){
      fprintf(stderr,"WARNING: EXPERIMENTAL: ellipticity: %.5e equivalent radii: r_a: %g r_b: %g\n",
	      E->data.ellipticity,E->data.ra,E->data.rc);
    }
    E->data.use_ellipse = 1;
  }else{
    E->data.ra = E->data.rc = E->data.efac=1.0;
    E->data.use_ellipse = 0;
  }
  /* 
     centrifugal ratio between \omega^2 a^3/GM, 3.46775e-3 for the
     Earth at present day
  */
  input_double("rotation_m",&E->data.rotm,"0.0",m);
  if(fabs(E->data.rotm) > 5e-7){
    /* J2 from flattening */
    E->data.j2 = 2./3.*E->data.ellipticity*(1.-E->data.ellipticity/2.)-
      E->data.rotm/3.*(1.-3./2.*E->data.rotm-2./7.*E->data.ellipticity);
    /* normalized gravity at the equator */
    E->data.ge = 1/(E->data.ra*E->data.ra)*(1+3./2.*E->data.j2-E->data.rotm);
    if(E->parallel.me==0)
      fprintf(stderr,"WARNING: rotational fraction m: %.5e J2: %.5e g_e: %g\n",
	      E->data.rotm,E->data.j2,E->data.ge);
    E->data.use_rotation_g = 1;
  }else{
    E->data.use_rotation_g = 0;
  }
#else
  if(fabs(ell_tmp) > 5e-7){
    myerror(E,"ellipticity not zero, but not compiled with ALLOW_ELLIPTICAL");
  }
#endif
  input_float("radius",&tmp,"6371e3.0",m);
  E->data.radius_km = tmp / 1e3;

  E->data.therm_cond = E->data.therm_diff * E->data.density * E->data.Cp;

  E->data.ref_temperature = E->control.Atemp * E->data.therm_diff
    * E->data.ref_viscosity
    / (E->data.density * E->data.grav_acc * E->data.therm_exp)
    / (E->data.radius_km * E->data.radius_km * E->data.radius_km * 1e9);

  output_common_input(E);
  h5input_params(E);
  phase_change_input(E);
  lith_age_input(E);

  tic_input(E);
  tracer_input(E);

  viscosity_input(E);		/* moved the viscosity input behind
				   the tracer input */

  (E->problem_settings)(E);
#ifdef USE_PETSC
  /* PETSc related flags */
  input_boolean("use_petsc",&E->control.use_petsc,"off",m);
  input_boolean("petsc_linear",&E->control.petsc_linear,"on",m);
  input_boolean("petsc_nonlinear",&E->control.petsc_nonlinear,"off",m);
  input_boolean("petsc_schur",&E->control.petsc_schur,"off",m);
  input_float("petsc_uzawa_tol", &E->control.petsc_uzawa_tol, "1e-6", m);
#endif
  check_settings_consistency(E);
  return;
}

/* Checking the consistency of input parameters */
void check_settings_consistency(struct All_variables *E)
{

    if (E->control.CONJ_GRAD) {
        /* conjugate gradient has only one level */
        if(E->mesh.levels != 1)
            myerror(E, "Conjugate gradient solver is used. 'levels' must be 1.\n");
    }
    else {
        /* multigrid solver needs two or more levels */
        if(E->mesh.levels < 2)
            myerror(E, "number of multigrid levels < 2\n");
        if(E->mesh.levels > MAX_LEVELS)
            myerror(E, "number of multigrid levels out of bound\n");
    }

    /* remove angular momentum/rigid rotation should only be done in free
       convection of global models. */
    if(E->sphere.caps == 12 &&
       (E->control.remove_angular_momentum || E->control.remove_rigid_rotation) &&
       (E->mesh.topvbc || E->mesh.botvbc || E->control.side_sbcs)) {
      if(E->parallel.me == 0)
	 fprintf(stderr,"\nWARNING: The input parameters impose boundary velocity, but also remove angular momentum/rigid rotation!\n\n");
    }

    /* no z_layer input found */
    if((fabs(E->viscosity.zbase_layer[0]+999) < 1e-5) &&
       (fabs(E->viscosity.zbase_layer[1]+999) < 1e-5)) {

        if(E->viscosity.num_mat != 4)
            myerror(E,"error: either use z_layer for non dim layer depths, or set num_mat to four");

        E->viscosity.zbase_layer[0] = E->viscosity.zlith;
        E->viscosity.zbase_layer[1] = E->viscosity.z410;
        E->viscosity.zbase_layer[2] = E->viscosity.zlm;
        E->viscosity.zbase_layer[3] = E->viscosity.zcmb; /* the lowest layers is never checked, really 
							    if x3 < zlm, then the last layers gets assigned
							    i left this in for backward compatibility
							 */
    }

    if (strcmp(E->output.vtk_format, "binary") == 0) {
#ifndef USE_GZDIR
        /* zlib is required for vtk binary output */
        if(E->parallel.me == 0) {
            fputs("VTK binary output requires zlib, but couldn't find it at 'configure' time. Please either use VTK ascii output or re-run 'configure' with suitable flags.\n", stderr);
        }
        parallel_process_termination();
#endif
    }

    return;
}


/* Setup global mesh parameters */
void global_derived_values(struct All_variables *E)
{
    int d,i,nox,noz,noy;

   E->mesh.levmax = E->mesh.levels-1;
   E->mesh.gridmax = E->mesh.levmax;

   E->mesh.elx = E->mesh.nox-1;
   E->mesh.ely = E->mesh.noy-1;
   E->mesh.elz = E->mesh.noz-1;

   if(E->sphere.caps == 1) {
       /* number of nodes, excluding overlaping nodes between processors */
       E->mesh.nno = E->sphere.caps * E->mesh.nox * E->mesh.noy * E->mesh.noz;
   }
   else {
       /* number of nodes, excluding overlaping nodes between processors */
       /* each cap has one row of nox and one row of noy overlapped, exclude these nodes.
        * nodes at north/south poles are exclued by all caps, include them by 2*noz*/
       E->mesh.nno = E->sphere.caps * (E->mesh.nox-1) * (E->mesh.noy-1) * E->mesh.noz
           + 2*E->mesh.noz;
   }

   E->mesh.nel = E->sphere.caps*E->mesh.elx*E->mesh.elz*E->mesh.ely;

   E->mesh.nnov = E->mesh.nno;

  /* this is a rough estimate for global neq, a more accurate neq will
     be computed later. */
   E->mesh.neq = E->mesh.nnov*E->mesh.nsd;

   E->mesh.npno = E->mesh.nel;
   E->mesh.nsf = E->mesh.nox*E->mesh.noy;

   for(i=E->mesh.levmax;i>=E->mesh.levmin;i--) {
      nox = E->mesh.mgunitx * (int) pow(2.0,(double)i)*E->parallel.nprocx + 1;
      noy = E->mesh.mgunity * (int) pow(2.0,(double)i)*E->parallel.nprocy + 1;
      noz = E->mesh.mgunitz * (int) pow(2.0,(double)i)*E->parallel.nprocz + 1;

      E->mesh.ELX[i] = nox-1;
      E->mesh.ELY[i] = noy-1;
      E->mesh.ELZ[i] = noz-1;
      if(E->sphere.caps == 1) {
          E->mesh.NNO[i] = nox * noz * noy;
      }
      else {
          E->mesh.NNO[i] = E->sphere.caps * (nox-1) * (noy-1) * noz + 2 * noz;
      }
      E->mesh.NEL[i] = E->sphere.caps * (nox-1) * (noz-1) * (noy-1);
      E->mesh.NPNO[i] = E->mesh.NEL[i] ;
      E->mesh.NOX[i] = nox;
      E->mesh.NOZ[i] = noz;
      E->mesh.NOY[i] = noy;

      E->mesh.NNOV[i] = E->mesh.NNO[i];
      E->mesh.NEQ[i] = E->mesh.nsd * E->mesh.NNOV[i] ;

   }

   /* Scaling from dimensionless units to Millions of years for input velocity
      and time, timdir is the direction of time for advection. CPC 6/25/00 */

    /* Myr */
    E->data.scalet = (E->data.radius_km*1e3*E->data.radius_km*1e3/E->data.therm_diff)/(1.e6*365.25*24*3600);
    /* cm/yr */
    E->data.scalev = (E->data.radius_km*1e3/E->data.therm_diff)/(100*365.25*24*3600);
    E->data.timedir = E->control.Atemp / fabs(E->control.Atemp);


    if(E->control.print_convergence && E->parallel.me==0) {
	fprintf(stderr,"Problem has %i x %i x %i nodes per cap, %i nodes and %i elements in total\n",
                E->mesh.nox, E->mesh.noz, E->mesh.noy, E->mesh.nno, E->mesh.nel);
	fprintf(E->fp,"Problem has %i x %i x %i nodes per cap, %i nodes and %i elements in total\n",
                E->mesh.nox, E->mesh.noz, E->mesh.noy, E->mesh.nno, E->mesh.nel);
    }
   return;
}


/* ===================================
   Functions which set up details
   common to all problems follow ...
   ===================================  */

void allocate_common_vars(E)
     struct All_variables *E;

{
    void set_up_nonmg_aliases();
    int m,n,snel,nsf,elx,ely,nox,noy,noz,nno,nel,npno,lim;
    int k,i,j,d,l,nno_l,npno_l,nozl,nnov_l,nxyz;

    m=0;
    n=1;

  npno = E->lmesh.npno;
  nel  = E->lmesh.nel;
  nno  = E->lmesh.nno;
  nsf  = E->lmesh.nsf;
  noz  = E->lmesh.noz;
  nox  = E->lmesh.nox;
  noy  = E->lmesh.noy;
  elx  = E->lmesh.elx;
  ely  = E->lmesh.ely;

  E->P        = (double *) malloc(npno*sizeof(double));
  E->T        = (double *) malloc((nno+1)*sizeof(double));
  E->NP       = (float *) malloc((nno+1)*sizeof(float));
  E->buoyancy[CPPR] = (double *) malloc((nno+1)*sizeof(double));

  E->gstress[CPPR] = (float *) malloc((6*nno+1)*sizeof(float));
  // TWB do we need this anymore XXX
  //E->stress[j]   = (float *) malloc((12*nsf+1)*sizeof(float));

  for(i=1;i<=E->mesh.nsd;i++)
      E->sphere.cap[CPPR].TB[i] = (float *)  malloc((nno+1)*sizeof(float));

  E->slice.tpg      = (float *)malloc((nsf+2)*sizeof(float));
  E->slice.tpgb     = (float *)malloc((nsf+2)*sizeof(float));
  E->slice.divg     = (float *)malloc((nsf+2)*sizeof(float));
  E->slice.vort     = (float *)malloc((nsf+2)*sizeof(float));
  E->slice.shflux    = (float *)malloc((nsf+2)*sizeof(float));
  E->slice.bhflux    = (float *)malloc((nsf+2)*sizeof(float));
  /*  if(E->mesh.topvbc==2 && E->control.pseudo_free_surf) */
  E->slice.freesurf    = (float *)malloc((nsf+2)*sizeof(float));

  E->mat = (int *) malloc((nel+2)*sizeof(int));
  E->VIP[CPPR] = (float *) malloc((nel+2)*sizeof(float));

  E->heating_adi    = (double *) malloc((nel+1)*sizeof(double));
  E->heating_visc   = (double *) malloc((nel+1)*sizeof(double));
  E->heating_latent = (double *) malloc((nel+1)*sizeof(double));

  /* lump mass matrix for the energy eqn */
  E->TMass = (double *) malloc((nno+1)*sizeof(double));

  /* nodal mass */
  E->NMass = (double *) malloc((nno+1)*sizeof(double));

  nxyz = max(nox*noz,nox*noy);
  nxyz = 2*max(nxyz,noz*noy);

  E->sien         = (struct SIEN *) malloc((nxyz+2)*sizeof(struct SIEN));
  E->surf_element = (int *) malloc((nxyz+2)*sizeof(int));
  E->surf_node    = (int *) malloc((nsf+2)*sizeof(int));


  /* density field */
  E->rho      = (double *) malloc((nno+1)*sizeof(double));

  /* horizontal average */
  E->Have.T         = (float *)malloc((E->lmesh.noz+2)*sizeof(float));
  E->Have.V[1]      = (float *)malloc((E->lmesh.noz+2)*sizeof(float));
  E->Have.V[2]      = (float *)malloc((E->lmesh.noz+2)*sizeof(float));

  E->sphere.gr = (double *)malloc((E->mesh.noz+1)*sizeof(double));

 for(i=E->mesh.levmin;i<=E->mesh.levmax;i++) {
  E->sphere.R[i] = (double *)  malloc((E->lmesh.NOZ[i]+1)*sizeof(double));
    nno  = E->lmesh.NNO[i];
    npno = E->lmesh.NPNO[i];
    nel  = E->lmesh.NEL[i];
    nox = E->lmesh.NOX[i];
    noz = E->lmesh.NOZ[i];
    noy = E->lmesh.NOY[i];
    elx = E->lmesh.ELX[i];
    ely = E->lmesh.ELY[i];
    snel=E->lmesh.SNEL[i];

    for(d=1;d<=E->mesh.nsd;d++)   {
      E->X[i][CPPR][d]  = (double *)  malloc((nno+1)*sizeof(double));
      E->SX[i][CPPR][d]  = (double *)  malloc((nno+1)*sizeof(double));
      }

    for(d=0;d<=3;d++)
      E->SinCos[i][CPPR][d]  = (double *)  malloc((nno+1)*sizeof(double));

    E->IEN[i][CPPR] = (struct IEN *)   malloc((nel+2)*sizeof(struct IEN));
    E->EL[i][CPPR]  = (struct SUBEL *) malloc((nel+2)*sizeof(struct SUBEL));
    E->sphere.area1[i][CPPR] = (double *) malloc((snel+1)*sizeof(double));
    for (k=1;k<=4;k++)
      E->sphere.angle1[i][CPPR][k] = (double *) malloc((snel+1)*sizeof(double));

    E->GNX[i][CPPR] = (struct Shape_function_dx *)malloc((nel+1)*sizeof(struct Shape_function_dx));
    E->GDA[i][CPPR] = (struct Shape_function_dA *)malloc((nel+1)*sizeof(struct Shape_function_dA));

    E->MASS[i]     = (double *) malloc((nno+1)*sizeof(double));
    E->ECO[i][CPPR] = (struct COORD *) malloc((nno+2)*sizeof(struct COORD));

    E->TWW[i][CPPR] = (struct FNODE *)   malloc((nel+2)*sizeof(struct FNODE));

    for(d=1;d<=E->mesh.nsd;d++)
      for(l=1;l<=E->lmesh.NNO[i];l++)  {
        E->SX[i][CPPR][d][l] = 0.0;
        E->X[i][CPPR][d][l] = 0.0;
        }

  }

 for(i=0;i<=E->output.llmax;i++)
  E->sphere.hindex[i] = (int *) malloc((E->output.llmax+3)
				       *sizeof(int));


 for(i=E->mesh.gridmin;i<=E->mesh.gridmax;i++) {

    nno  = E->lmesh.NNO[i];
    npno = E->lmesh.NPNO[i];
    nel  = E->lmesh.NEL[i];
    nox = E->lmesh.NOX[i];
    noz = E->lmesh.NOZ[i];
    noy = E->lmesh.NOY[i];
    elx = E->lmesh.ELX[i];
    ely = E->lmesh.ELY[i];

    nxyz = elx*ely;
    E->CC[i][CPPR] =(struct CC *)  malloc((1)*sizeof(struct CC));
    E->CCX[i][CPPR]=(struct CCX *)  malloc((1)*sizeof(struct CCX));

    E->elt_del[i][CPPR] = (struct EG *) malloc((nel+1)*sizeof(struct EG));

    if(E->control.inv_gruneisen != 0)
        E->elt_c[i][CPPR] = (struct EC *) malloc((nel+1)*sizeof(struct EC));

    E->EVI[i] = (float *) malloc((nel+1)*vpoints[E->mesh.nsd]*sizeof(float));
    E->BPI[i] = (double *) malloc((npno+1)*sizeof(double));

    E->ID[i][CPPR]  = (struct ID *)    malloc((nno+1)*sizeof(struct ID));
    E->VI[i]  = (float *)        malloc((nno+1)*sizeof(float));
    E->NODE[i][CPPR] = (unsigned int *)malloc((nno+1)*sizeof(unsigned int));

    nxyz = max(nox*noz,nox*noy);
    nxyz = 2*max(nxyz,noz*noy);
    nozl = max(noy,nox*2);



    E->parallel.EXCHANGE_sNODE[i][CPPR] = (struct PASS *) malloc((nozl+2)*sizeof(struct PASS));
    E->parallel.NODE[i][CPPR]   = (struct BOUND *) malloc((nxyz+2)*sizeof(struct BOUND));
    E->parallel.EXCHANGE_NODE[i][CPPR]= (struct PASS *) malloc((nxyz+2)*sizeof(struct PASS));
    E->parallel.EXCHANGE_ID[i][CPPR] = (struct PASS *) malloc((nxyz*E->mesh.nsd+3)*sizeof(struct PASS));

    for(l=1;l<=E->lmesh.NNO[i];l++)  {
      E->NODE[i][CPPR][l] = (INTX | INTY | INTZ);  /* and any others ... */
      E->VI[i][l] = 1.0;
      }


    }         /* end for cap and i & j  */

#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
 if(E->viscosity.allow_anisotropic_viscosity){ /* any anisotropic
						  viscosity */
   for(i=E->mesh.gridmin;i<=E->mesh.gridmax;i++) {
       nel  = E->lmesh.NEL[i];
       nno  = E->lmesh.NNO[i];
       E->EVI2[i][CPPR] = (float *) malloc((nel+1)*vpoints[E->mesh.nsd]*sizeof(float));
       E->avmode[i][CPPR] = (unsigned char *) malloc((nel+1)*vpoints[E->mesh.nsd]*sizeof(unsigned char));
       E->EVIn1[i][CPPR] = (float *) malloc((nel+1)*vpoints[E->mesh.nsd]*sizeof(float));
       E->EVIn2[i][CPPR] = (float *) malloc((nel+1)*vpoints[E->mesh.nsd]*sizeof(float));
       E->EVIn3[i][CPPR] = (float *) malloc((nel+1)*vpoints[E->mesh.nsd]*sizeof(float));
       
       E->VI2[i][CPPR]  = (float *)        malloc((nno+1)*sizeof(float));
       E->VIn1[i][CPPR]  = (float *)        malloc((nno+1)*sizeof(float));
       E->VIn2[i][CPPR]  = (float *)        malloc((nno+1)*sizeof(float));
       E->VIn3[i][CPPR]  = (float *)        malloc((nno+1)*sizeof(float));
       if((!(E->EVI2[i][CPPR]))||(!(E->VI2[i][CPPR]))||
	  (!(E->EVIn1[i][CPPR]))||(!(E->EVIn2[i][CPPR]))||(!(E->EVIn3[i][CPPR]))||
	  (!(E->VIn1[i][CPPR]))||(!(E->VIn2[i][CPPR]))||(!(E->VIn3[i][CPPR]))){
	 fprintf(stderr, "Error: Cannot allocate anisotropic visc memory, rank=%i\n",
		 E->parallel.me);
	 parallel_process_termination();
       }
     }
   E->viscosity.anisotropic_viscosity_init = FALSE;
 }
#endif


  for(k=1;k<=E->mesh.nsd;k++)
    for(i=1;i<=E->lmesh.nno;i++)
      E->sphere.cap[CPPR].TB[k][i] = 0.0;

  for(i=1;i<=E->lmesh.nno;i++)
     E->T[i] = 0.0;

  for(i=1;i<=E->lmesh.nel;i++)   {
      E->mat[i]=1;
      E->VIP[CPPR][i]=1.0;

      E->heating_adi[i] = 0;
      E->heating_visc[i] = 0;
      E->heating_latent[i] = 1.0;
  }

  for(i=0;i<E->lmesh.npno;i++)
      E->P[i] = 0.0;

  mat_prop_allocate(E);
  phase_change_allocate(E);
  set_up_nonmg_aliases(E);


  if (strcmp(E->output.format, "hdf5") == 0)
      h5output_allocate_memory(E);
}

/*  =========================================================  */

void allocate_velocity_vars(E)
     struct All_variables *E;

{
    int m,n,i,j,k,l;

    E->monitor.incompressibility = 0;
    E->monitor.fdotf = 0;
    E->monitor.vdotv = 0;
    E->monitor.pdotp = 0;

 m=0;
 n=1;
    E->lmesh.nnov = E->lmesh.nno;
    E->lmesh.neq = E->lmesh.nnov * E->mesh.nsd;

    E->temp = (double *) malloc((E->lmesh.neq+1)*sizeof(double));
    E->temp1 = (double *) malloc(E->lmesh.neq*sizeof(double));
    E->F = (double *) malloc(E->lmesh.neq*sizeof(double));
    E->U = (double *) malloc(E->lmesh.neq*sizeof(double));
    E->u1 = (double *) malloc(E->lmesh.neq*sizeof(double));


    for(i=1;i<=E->mesh.nsd;i++) {
      E->sphere.cap[CPPR].V[i] = (float *) malloc((E->lmesh.nnov+1)*sizeof(float));
      E->sphere.cap[CPPR].VB[i] = (float *)malloc((E->lmesh.nnov+1)*sizeof(float));
      E->sphere.cap[CPPR].Vprev[i] = (float *) malloc((E->lmesh.nnov+1)*sizeof(float));
    }

    for(i=0;i<E->lmesh.neq;i++)
      E->U[i] = E->temp[i] = E->temp1[i] = 0.0;


    for(k=1;k<=E->mesh.nsd;k++)
      for(i=1;i<=E->lmesh.nnov;i++)
        E->sphere.cap[CPPR].VB[k][i] = 0.0;


  for(l=E->mesh.gridmin;l<=E->mesh.gridmax;l++) {
      E->lmesh.NEQ[l] = E->lmesh.NNOV[l] * E->mesh.nsd;

      E->BI[l] = (double *) malloc((E->lmesh.NEQ[l])*sizeof(double));
      k = (E->lmesh.NOX[l]*E->lmesh.NOZ[l]+E->lmesh.NOX[l]*E->lmesh.NOY[l]+
          E->lmesh.NOY[l]*E->lmesh.NOZ[l])*6;
      E->zero_resid[l][CPPR] = (int *) malloc((k+2)*sizeof(int));
      E->parallel.Skip_id[l][CPPR] = (int *) malloc((k+2)*sizeof(int));

      for(i=0;i<E->lmesh.NEQ[l];i++) {
         E->BI[l][i]=0.0;
         }

      }   /* end for j & l */
}


/*  =========================================================  */

void global_default_values(E)
     struct All_variables *E;
{

  /* FIRST: values which are not changed routinely by the user */

  E->control.v_steps_low = 10;
  E->control.v_steps_upper = 1;
  E->control.accuracy = 1.0e-4;
  E->control.verbose=0; /* debugging/profiles */

  /* SECOND: values for which an obvious default setting is useful */

    E->control.stokes=0;
    E->control.restart=0;
    E->control.CONVECTION = 0;
    E->control.CART2D = 0;
    E->control.CART3D = 0;
    E->control.CART2pt5D = 0;
    E->control.AXI = 0;
    E->control.CONJ_GRAD = 0;
    E->control.NMULTIGRID = 0;
    E->control.augmented_Lagr = 0;
    E->control.augmented = 0.0;

    E->trace.fpt = NULL;
    E->control.tracer = 0;
    E->composition.on = 0;

  E->parallel.nprocx=1; E->parallel.nprocz=1; E->parallel.nprocy=1;

  E->mesh.levmax=0;
  E->mesh.levmin=0;
  E->mesh.gridmax=0;
  E->mesh.gridmin=0;
  E->mesh.noz = 1;    E->mesh.nzs = 1;  E->lmesh.noz = 1;    E->lmesh.nzs = 1;
  E->mesh.noy = 1;    E->mesh.nys = 1;  E->lmesh.noy = 1;    E->lmesh.nys = 1;
  E->mesh.nox = 1;    E->mesh.nxs = 1;  E->lmesh.nox = 1;    E->lmesh.nxs = 1;

  E->sphere.ro = 1.0;
  E->sphere.ri = 0.5;

  E->control.precondition = 0;  /* for larger visc contrasts turn this back on  */

  E->mesh.toptbc = 1; /* fixed t */
  E->mesh.bottbc = 1;
  E->mesh.topvbc = 0; /* stress */
  E->mesh.botvbc = 0;
  E->control.VBXtopval=0.0;
  E->control.VBYtopval=0.0;
  E->control.VBXbotval=0.0;
  E->control.VBYbotval=0.0;

  E->data.radius_km = 6370.0; /* Earth, whole mantle defaults */
  E->data.grav_acc = 9.81;
  E->data.therm_diff = 1.0e-6;
  E->data.therm_exp = 3.e-5;
  E->data.density = 3300.0;
  E->data.ref_viscosity=1.e21;
  E->data.density_above = 1000.0;    /* sea water */
  E->data.density_below = 6600.0;    /* sea water */

  E->data.Cp = 1200.0;
  E->data.therm_cond = 3.168;
  E->data.res_density = 3300.0;  /* density when X = ... */
  E->data.res_density_X = 0.3;
  E->data.melt_density = 2800.0;
  E->data.permeability = 3.0e-10;
  E->data.gas_const = 8.3;
  E->data.surf_heat_flux = 4.4e-2;

  E->data.grav_const = 6.6742e-11;

  E->data.youngs_mod = 1.0e11;
  E->data.Te = 0.0;
  E->data.T_sol0 = 1373.0;      /* Dave's values 1991 (for the earth) */
  E->data.Tsurf = 273.0;
  E->data.dTsol_dz = 3.4e-3 ;
  E->data.dTsol_dF = 440.0;
  E->data.dT_dz = 0.48e-3;
  E->data.delta_S = 250.0;
  E->data.ref_temperature = 2 * 1350.0; /* fixed temperature ... delta T */

  /* THIRD: you forgot and then went home, let's see if we can help out */

    sprintf(E->control.data_prefix,"citcom.tmp.%d",getpid());

    E->control.NASSEMBLE = 0;

    E->monitor.elapsed_time=0.0;

    E->control.record_all_until = 10000000;

  return;
}


/* =============================================================
   ============================================================= */

void check_bc_consistency(E)
     struct All_variables *E;

{ 
  int i,j,lev;

    for(i=1;i<=E->lmesh.nno;i++) {
      if ((E->node[i] & VBX) && (E->node[i] & SBX))
        printf("Inconsistent x velocity bc at %d\n",i);
      if ((E->node[i] & VBZ) && (E->node[i] & SBZ))
        printf("Inconsistent z velocity bc at %d\n",i);
      if ((E->node[i] & VBY) && (E->node[i] & SBY))
        printf("Inconsistent y velocity bc at %d\n",i);
      if ((E->node[i] & TBX) && (E->node[i] & FBX))
        printf("Inconsistent x temperature bc at %d\n",i);
      if ((E->node[i] & TBZ) && (E->node[i] & FBZ))
        printf("Inconsistent z temperature bc at %d\n",i);
      if ((E->node[i] & TBY) && (E->node[i] & FBY))
        printf("Inconsistent y temperature bc at %d\n",i);
      }

  for(lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++) {
      for(i=1;i<=E->lmesh.NNO[lev];i++) {
        if ((E->NODE[lev][CPPR][i] & VBX) && (E->NODE[lev][CPPR][i]  & SBX))
          printf("Inconsistent x velocity bc at %d,%d\n",lev,i);
        if ((E->NODE[lev][CPPR][i] & VBZ) && (E->NODE[lev][CPPR][i]  & SBZ))
          printf("Inconsistent z velocity bc at %d,%d\n",lev,i);
        if ((E->NODE[lev][CPPR][i] & VBY) && (E->NODE[lev][CPPR][i]  & SBY))
          printf("Inconsistent y velocity bc at %d,%d\n",lev,i);
        /* Tbc's not applicable below top level */
        }
    }   /* end for  j and lev */
}

void set_up_nonmg_aliases(struct All_variables *E)
{ /* Aliases for functions only interested in the highest mg level */

  int i;

  E->eco = E->ECO[E->mesh.levmax][CPPR];
  E->ien = E->IEN[E->mesh.levmax][CPPR];
  E->id = E->ID[E->mesh.levmax][CPPR];
  E->Vi = E->VI[E->mesh.levmax];
  E->EVi = E->EVI[E->mesh.levmax];
  E->node = E->NODE[E->mesh.levmax][CPPR];
  E->cc = E->CC[E->mesh.levmax][CPPR];
  E->ccx = E->CCX[E->mesh.levmax][CPPR];
  E->Mass = E->MASS[E->mesh.levmax];
  E->gDA[CPPR] = E->GDA[E->mesh.levmax][CPPR];
  E->gNX[CPPR] = E->GNX[E->mesh.levmax][CPPR];

  for (i=1;i<=E->mesh.nsd;i++)    {
    E->x[i] = E->X[E->mesh.levmax][CPPR][i];
    E->sx[i] = E->SX[E->mesh.levmax][CPPR][i];
    }
}

void report(E,string)
     struct All_variables *E;
     char * string;
{ if(E->control.verbose && E->parallel.me==0)
    { fprintf(stderr,"%s\n",string);
      fflush(stderr);
    }
}

void record(E,string)
     struct All_variables *E;
     char * string;
{ if(E->control.verbose && E->fp)
    { fprintf(E->fp,"%s\n",string);
      fflush(E->fp);
    }

  return;
}



/* =============================================================
   Initialize values which are not problem dependent.
   NOTE: viscosity may be a function of all previous
   input fields (temperature, pressure, velocity, chemistry) and
   so is always to be done last.
   ============================================================= */


/* This function is replaced by CitcomS.Components.IC.launch()*/
void common_initial_fields(E)
    struct All_variables *E;
{
    void initial_pressure();
    void initial_velocity();
    void initial_viscosity();

    initial_pressure(E);
    initial_velocity(E);
    initial_viscosity(E);

    return;

}

/* ========================================== */

void initial_pressure(E)
     struct All_variables *E;
{
    int i,m;
    report(E,"Initialize pressure field");

    for(i=0;i<E->lmesh.npno;i++)
      E->P[i]=0.0;
}

void initial_velocity(E)
     struct All_variables *E;
{
    int i,m;
    report(E,"Initialize velocity field");

    for(i=1;i<=E->lmesh.nnov;i++)   {
        E->sphere.cap[CPPR].V[1][i]=0.0;
        E->sphere.cap[CPPR].V[2][i]=0.0;
        E->sphere.cap[CPPR].V[3][i]=0.0;
    }
}



static void open_log(struct All_variables *E)
{
  char logfile[255];

  E->fp = NULL;
  if (strcmp(E->output.format, "ascii-gz") == 0)
    sprintf(logfile,"%s/log", E->control.data_dir);
  else
    sprintf(logfile,"%s.log", E->control.data_file);

  if (E->control.restart || E->control.post_p)
      /* append the log file if restart */
      E->fp = output_open(logfile, "a");
  else
      E->fp = output_open(logfile, "w");

  return;
}


static void open_time(struct All_variables *E)
{
  char timeoutput[255];

  E->fptime = NULL;
  if (E->parallel.me == 0) {
  if (strcmp(E->output.format, "ascii-gz") == 0)
    sprintf(timeoutput,"%s/time", E->control.data_dir);
  else
    sprintf(timeoutput,"%s.time", E->control.data_file);

  if (E->control.restart || E->control.post_p)
      /* append the time file if restart */
      E->fptime = output_open(timeoutput, "a");
  else
      E->fptime = output_open(timeoutput, "w");
  }

  return;
}


static void open_info(struct All_variables *E)
{
  char output_file[255];

  E->fp_out = NULL;
  if (E->control.verbose) {
  if (strcmp(E->output.format, "ascii-gz") == 0)
    sprintf(output_file,"%s/info.%d", E->control.data_dir, E->parallel.me);
  else
    sprintf(output_file,"%s.info.%d", E->control.data_file, E->parallel.me);
  E->fp_out = output_open(output_file, "w");
  }

  return;
}

void open_qfiles(struct All_variables *E) /* additional heat
					     flux output */
{
  char output_file[255];

  /* only one CPU will write to those */
  if((E->parallel.me_loc[3] == E->parallel.nprocz-1) &&
     (E->parallel.me==E->parallel.nprocz-1)){
    /* top heat flux and other stat quantities */
    if (strcmp(E->output.format, "ascii-gz") == 0)
      sprintf(output_file,"%s/qt.dat", E->control.data_dir);
    else
      sprintf(output_file,"%s.qt.dat", E->control.data_file);
    if(E->control.restart)
      E->output.fpqt = output_open(output_file, "a"); /* append for restart */
    else
      E->output.fpqt = output_open(output_file, "w");
  }else{
    E->output.fpqt = NULL;
  }
  if (E->parallel.me_loc[3] == 0)    {
    /* bottom heat flux and other stat quantities */
    if (strcmp(E->output.format, "ascii-gz") == 0)
      sprintf(output_file,"%s/qb.dat", E->control.data_dir);
    else
      sprintf(output_file,"%s.qb.dat", E->control.data_file);
    if(E->control.restart)
      E->output.fpqb = output_open(output_file, "a"); /* append */
    else
      E->output.fpqb = output_open(output_file, "w");
  }else{
    E->output.fpqb = NULL;
  }


  return;
}


static void output_parse_optional(struct  All_variables *E)
{
    char* strip(char*);

    int pos, len;
    char *prev, *next;

    len = strlen(E->output.optional);
    /* fprintf(stderr, "### length of optional is %d\n", len); */
    pos = 0;
    next = E->output.optional;

    E->output.connectivity = 0;
    E->output.stress = 0;
    E->output.pressure = 0;
    E->output.surf = 0;
    E->output.botm = 0;
    E->output.geoid = 0;
    E->output.horiz_avg = 0;
    E->output.seismic = 0;
    E->output.coord_bin = 0;
    E->output.tracer = 0;
    E->output.comp_el = 0;
    E->output.comp_nd = 0;
    E->output.heating = 0;

    while(1) {
        /* get next field */
        prev = strsep(&next, ",");

        /* break if no more field */
        if(prev == NULL) break;

        /* skip if empty */
        if(prev[0] == '\0') continue;

        /* strip off leading and trailing whitespaces */
        prev = strip(prev);

        /* skip empty field */
        if (strlen(prev) == 0) continue;

        /* fprintf(stderr, "### %s: %s\n", prev, next); */
        if(strcmp(prev, "connectivity")==0)
            E->output.connectivity = 1;
        else if(strcmp(prev, "stress")==0)
            E->output.stress = 1;
        else if(strcmp(prev, "pressure")==0)
            E->output.pressure = 1;
        else if(strcmp(prev, "surf")==0)
            E->output.surf = 1;
        else if(strcmp(prev, "botm")==0)
            E->output.botm = 1;
        else if(strcmp(prev, "geoid")==0)
	    if (E->parallel.nprocxy != 12) {
		fprintf(stderr, "Warning: geoid calculation only works in full version. Disabled\n");
	    }
	    else {
		/* geoid calculation requires surface and CMB topo */
		/* make sure the topos are available!              */
		E->output.geoid  = 1;
	    }
        else if(strcmp(prev, "horiz_avg")==0)
            E->output.horiz_avg = 1;
        else if(strcmp(prev, "seismic")==0) {
            E->output.seismic = E->output.coord_bin = 1;

            /* Total temperature contrast is important when computing seismic velocity,
             * but it is derived from several parameters. Output it clearly. */
            if(E->parallel.me==0) {
                fprintf(stderr, "Total temperature contrast = %f K\n", E->data.ref_temperature);
                fprintf(E->fp, "Total temperature contrast = %f K\n", E->data.ref_temperature);
            }
        }
        else if(strcmp(prev, "tracer")==0)
            E->output.tracer = 1;
        else if(strcmp(prev, "comp_el")==0)
            E->output.comp_el = 1;
        else if(strcmp(prev, "comp_nd")==0)
            E->output.comp_nd = 1;
        else if(strcmp(prev, "heating")==0)
            E->output.heating = 1;
        else
            if(E->parallel.me == 0)
                fprintf(stderr, "Warning: unknown field for output_optional: %s\n", prev);

    }

    return;
}

/* check whether E->control.data_file contains a path seperator */
static void chk_prefix(struct  All_variables *E)
{
  char *found;

  found = strchr(E->control.data_prefix, '/');
  if (found) {
      fprintf(stderr, "error in input parameter: datafile='%s' contains '/'\n", E->control.data_file);
      parallel_process_termination();
  }

  if (E->control.restart || E->control.post_p ||
      (E->convection.tic_method == -1) ||
      (E->control.tracer && (E->trace.ic_method == 2))) {
      found = strchr(E->control.data_prefix_old, '/');
      if (found) {
	  fprintf(stderr, "error in input parameter: datafile_old='%s' contains '/'\n", E->control.data_file);
	  parallel_process_termination();
      }
  }
}


/* search src and substitue the 1st occurance of target by value */
static void expand_str(char *src, size_t max_size,
		       const char *target, const char *value)
{
    char *pos, *end, *new_end;
    size_t end_len, value_len;

    /* is target a substring of src? */
    pos = strstr(src, target);
    if (pos != NULL) {
        value_len = strlen(value);

	/* the end part of the original string... */
	end = pos + strlen(target);
        /* ...and where it is going */
        new_end = pos + value_len;
        end_len = strlen(end);
        if (new_end + end_len >= src + max_size) {
            /* too long */
            return;
        }

	/* move the end part of the original string */
        memmove(new_end, end, end_len + 1); /* incl. null byte */

        /* insert the value */
        memcpy(pos, value, value_len);
    }
}

static void expand_datadir(struct All_variables *E, char *datadir)
{
    char *found, *err;
    char tmp[150];
    int diff;
    FILE *pipe;
    const char str1[] = "%HOSTNAME";
    const char str2[] = "%RANK";
    const char str3[] = "%DATADIR";
    const char str3_prog[] = "citcoms_datadir";

    /* expand str1 by machine's hostname */
    found = strstr(datadir, str1);
    if (found) {
	gethostname(tmp, 100);
	expand_str(datadir, 150, str1, tmp);
    }

    /* expand str2 by MPI rank */
    found = strstr(datadir, str2);
    if (found) {
	sprintf(tmp, "%d", E->parallel.me);
	expand_str(datadir, 150, str2, tmp);
    }

    /* expand str3 by the result of the external program */
    diff = strcmp(datadir, str3);
    if (!diff) {
	pipe = popen(str3_prog, "r");
	err = fgets(tmp, 150, pipe);
	pclose(stdout);
	if (err != NULL)
	    sscanf(tmp, " %s", datadir);
	else {
	    fprintf(stderr, "Cannot get datadir from command '%s'\n", str3_prog);
	    parallel_process_termination();
	}
    }
}


void mkdatadir(const char *dir)
{
  int err;

  err = mkdir(dir, 0755);
  if (err && errno != EEXIST) {
      /* if error occured and the directory is not exisitng */
      fprintf(stderr, "Cannot make new directory '%s'\n", dir);
      parallel_process_termination();
  }
}


void output_init(struct  All_variables *E)
{
    chk_prefix(E);
    expand_datadir(E, E->control.data_dir);
    mkdatadir(E->control.data_dir);
    snprintf(E->control.data_file, 200, "%s/%s", E->control.data_dir,
	     E->control.data_prefix);

    if (E->control.restart || E->control.post_p ||
        (E->convection.tic_method == -1) ||
        (E->control.tracer && (E->trace.ic_method == 2))) {
	expand_datadir(E, E->control.data_dir_old);
	snprintf(E->control.old_P_file, 200, "%s/%s", E->control.data_dir_old,
		 E->control.data_prefix_old);
    }

    open_log(E);
    open_time(E);
    open_info(E);

    if (strcmp(E->output.format, "ascii") == 0) {
        E->problem_output = output;
    }
    else if (strcmp(E->output.format, "hdf5") == 0)
        E->problem_output = h5output;
    else if (strcmp(E->output.format, "vtk") == 0)
        E->problem_output = vtk_output;
#ifdef USE_GZDIR
    else if (strcmp(E->output.format, "ascii-gz") == 0)
        E->problem_output = gzdir_output;
    else {
        /* indicate error here */
        if (E->parallel.me == 0) {
            fprintf(stderr, "wrong output_format, must be 'ascii', 'hdf5', 'ascii-gz' or 'vtk'\n");
            fprintf(E->fp, "wrong output_format, must be  'ascii', 'hdf5' 'ascii-gz', or 'vtk'\n");
        }
        parallel_process_termination();
    }
#else
    else {
        /* indicate error here */
        if (E->parallel.me == 0) {
            fprintf(stderr, "wrong output_format, must be 'ascii', 'hdf5', or 'vtk' (USE_GZDIR undefined)\n");
            fprintf(E->fp, "wrong output_format, must be 'ascii', 'hdf5', or 'vtk' (USE_GZDIR undefined)\n");
        }
        parallel_process_termination();
    }
#endif

    output_parse_optional(E);
}



void output_finalize(struct  All_variables *E)
{
  char message[255],files[255];
  if (E->fp)
    fclose(E->fp);
  if (E->fptime)
    fclose(E->fptime);
  if (E->fp_out)
    fclose(E->fp_out);
  if (E->trace.fpt)
    fclose(E->trace.fpt);
  if(E->output.fpqt)
    fclose(E->output.fpqt);
  if(E->output.fpqb)
    fclose(E->output.fpqb);

#ifdef USE_GZDIR
  /*
     remove VTK geo file in case we used that for IO
  */
  if((E->output.gzdir.vtk_io != 0) &&
     (strcmp(E->output.format, "ascii-gz") == 0)){
    if((E->output.gzdir.vtk_io == 3)||(E->parallel.me == 0)){
      /* delete the geo files */
      get_vtk_filename(files,1,E,0);
      remove(files);
      if(E->parallel.me == 0){
	/* close the log */
	if(E->output.gzdir.vtk_fp)
	  fclose(E->output.gzdir.vtk_fp);
      }
    }
  }
#endif
}


char* strip(char *input)
{
    int end;
    char *str;
    end = strlen(input) - 1;
    str = input;

    /* trim trailing whitespace */
    while (isspace(str[end]))
        end--;

    str[++end] = 0;

    /* trim leading whitespace */
    while(isspace(*str))
        str++;

    return str;
}


