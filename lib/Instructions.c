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
#include "interuption.h"

void parallel_process_termination();
void allocate_common_vars(struct All_variables*);
void allocate_velocity_vars(struct All_variables*);
void check_bc_consistency(struct All_variables*);
void construct_elt_gs(struct All_variables*);
void construct_elt_cs(struct All_variables*);
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
void get_vtk_filename(char *,int,struct All_variables *,int);
void myerror(struct All_variables *,char *);
void open_qfiles(struct All_variables *) ;


void initial_mesh_solver_setup(struct All_variables *E)
{

    E->monitor.cpu_time_at_last_cycle =
        E->monitor.cpu_time_at_start = CPU_time0();

    output_init(E);
    (E->problem_derived_values)(E);   /* call this before global_derived_  */
    (E->solver.global_derived_values)(E);

    (E->solver.parallel_processor_setup)(E);   /* get # of proc in x,y,z */
    (E->solver.parallel_domain_decomp0)(E);  /* get local nel, nno, elx, nox et al */

    allocate_common_vars(E);
    (E->problem_allocate_vars)(E);
    (E->solver_allocate_vars)(E);

           /* logical domain */
    construct_ien(E);
    construct_surface(E);
    (E->solver.construct_boundary)(E);
    (E->solver.parallel_domain_boundary_nodes)(E);

           /* physical domain */
    (E->solver.node_locations)(E);

    allocate_velocity_vars(E);

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

    construct_masks(E);		/* order is important here */
    construct_id(E);
    construct_lm(E);

    (E->solver.parallel_communication_routs_v)(E);
    (E->solver.parallel_communication_routs_s)(E);

    construct_sub_element(E);
    construct_shape_functions(E);
    construct_elt_gs(E);
    if(E->control.inv_gruneisen != 0)
        construct_elt_cs(E);

    reference_state(E);

    /* this matrix results from spherical geometry */
    /* construct_c3x3matrix(E); */

    mass_matrix(E);

    construct_surf_det (E);
    construct_bdry_det (E);

    set_sphere_harmonics (E);

    if(E->control.tracer) {
	tracer_initial_settings(E);
	(E->problem_tracer_setup)(E);
    }

}


void read_instructions(struct All_variables *E, char *filename)
{
    void read_initial_settings();
    void global_default_values();

    void setup_parser();
    void shutdown_parser();

    /* =====================================================
       Global interuption handling routine defined once here
       =====================================================  */

    set_signal();

    /* ==================================================
       Initialize from the command line
       from startup files. (See Parsing.c).
       ==================================================  */

    setup_parser(E,filename);

    global_default_values(E);
    read_initial_settings(E);

    shutdown_parser(E);

    return;
}


/* This function is replaced by CitcomS.Solver._setup() */
void initial_setup(struct All_variables *E)
{
    void general_stokes_solver_setup();
    void initial_mesh_solver_setup();

    initial_mesh_solver_setup(E);

    general_stokes_solver_setup(E);

    (E->next_buoyancy_field_init)(E);


    if (E->parallel.me==0) fprintf(stderr,"time=%f\n",
                                   CPU_time0()-E->monitor.cpu_time_at_start);

    return;
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
  int m=E->parallel.me;

  /* first the problem type (defines subsequent behaviour) */

  input_string("Problem",E->control.PROBLEM_TYPE,"convection",m);
  if ( strcmp(E->control.PROBLEM_TYPE,"convection") == 0)  {
    E->control.CONVECTION = 1;
    set_convection_defaults(E);
  }

  else if ( strcmp(E->control.PROBLEM_TYPE,"convection-chemical") == 0) {
    E->control.CONVECTION = 1;
    set_convection_defaults(E);
  }

  else {
    fprintf(E->fp,"Unable to determine problem type, assuming convection ... \n");
    E->control.CONVECTION = 1;
    set_convection_defaults(E);
  }

  input_string("Geometry",E->control.GEOMETRY,"sphere",m);
  if ( strcmp(E->control.GEOMETRY,"cart2d") == 0)
    { E->control.CART2D = 1;
    (E->solver.set_2dc_defaults)(E);}
  else if ( strcmp(E->control.GEOMETRY,"axi") == 0)
    { E->control.AXI = 1;
    }
  else if ( strcmp(E->control.GEOMETRY,"cart2pt5d") == 0)
    { E->control.CART2pt5D = 1;
    (E->solver.set_2pt5dc_defaults)(E);}
  else if ( strcmp(E->control.GEOMETRY,"cart3d") == 0)
    { E->control.CART3D = 1;
    (E->solver.set_3dc_defaults)(E);}
  else if ( strcmp(E->control.GEOMETRY,"sphere") == 0)
    {
      (E->solver.set_3dsphere_defaults)(E);}
  else
    { fprintf(E->fp,"Unable to determine geometry, assuming cartesian 2d ... \n");
    E->control.CART2D = 1;
    (E->solver.set_2dc_defaults)(E); }

  input_string("Solver",E->control.SOLVER_TYPE,"cgrad",m);
  if ( strcmp(E->control.SOLVER_TYPE,"cgrad") == 0)
    { E->control.CONJ_GRAD = 1;
    set_cg_defaults(E);}
  else if ( strcmp(E->control.SOLVER_TYPE,"multigrid") == 0)
    { E->control.NMULTIGRID = 1;
    set_mg_defaults(E);}
  else if ( strcmp(E->control.SOLVER_TYPE,"multigrid-el") == 0)
    { E->control.EMULTIGRID = 1;
    set_mg_defaults(E);}
  else
    { if (E->parallel.me==0) fprintf(stderr,"Unable to determine how to solve, specify Solver=VALID_OPTION \n");
    parallel_process_termination();
    }


  /* admin */

  input_string("Spacing",E->control.NODE_SPACING,"regular",m);
  if ( strcmp(E->control.NODE_SPACING,"regular") == 0)
    E->control.GRID_TYPE = 1;
  else if ( strcmp(E->control.NODE_SPACING,"bound_lyr") == 0)
    E->control.GRID_TYPE = 2;
  else if ( strcmp(E->control.NODE_SPACING,"region") == 0)
    E->control.GRID_TYPE = 3;
  else if ( strcmp(E->control.NODE_SPACING,"ortho_files") == 0)
    E->control.GRID_TYPE = 4;
  else
    {  E->control.GRID_TYPE = 1; }

  /* Information on which files to print, which variables of the flow to calculate and print.
     Default is no information recorded (apart from special things for given applications.
  */

  input_string("datadir",E->control.data_dir,".",m);
  input_string("datafile",E->control.data_prefix,"initialize",m);
  input_string("datadir_old",E->control.data_dir_old,".",m);
  input_string("datafile_old",E->control.data_prefix_old,"initialize",m);

  input_int("mgunitx",&(E->mesh.mgunitx),"1",m);
  input_int("mgunitz",&(E->mesh.mgunitz),"1",m);
  input_int("mgunity",&(E->mesh.mgunity),"1",m);
  input_int("levels",&(E->mesh.levels),"0",m);

  input_int("coor",&(E->control.coor),"0",m);
  if(E->control.coor == 2){	/* refinement */
    E->control.coor_refine[0] = 0.10; /* bottom 10% */
    E->control.coor_refine[1] = 0.15; /* get 15% of the nodes */
    E->control.coor_refine[2] = 0.10; /* top 10% */
    E->control.coor_refine[3] = 0.20; /* get 20% of the nodes */
    input_float_vector("coor_refine",4,E->control.coor_refine,m);
  }
  input_string("coor_file",E->control.coor_file,"",m);

  input_int("nprocx",&(E->parallel.nprocx),"1",m);
  input_int("nprocy",&(E->parallel.nprocy),"1",m);
  input_int("nprocz",&(E->parallel.nprocz),"1",m);
  input_int("nproc_surf",&(E->parallel.nprocxy),"1",m);


  input_boolean("node_assemble",&(E->control.NASSEMBLE),"off",m);
  /* general mesh structure */

  input_boolean("verbose",&(E->control.verbose),"off",m);
  input_boolean("see_convergence",&(E->control.print_convergence),"off",m);

  input_int("stokes_flow_only",&(E->control.stokes),"0",m);

  input_int("restart",&(E->control.restart),"0",m);
  input_int("post_p",&(E->control.post_p),"0",m);
  input_int("solution_cycles_init",&(E->monitor.solution_cycles_init),"0",m);

  /* for layers    */
  input_float("z_cmb",&(E->viscosity.zcmb),"0.45",m);
  input_float("z_lmantle",&(E->viscosity.zlm),"0.45",m);
  input_float("z_410",&(E->viscosity.z410),"0.225",m);
  input_float("z_lith",&(E->viscosity.zlith),"0.225",m);

  /*  the start age and initial subduction history   */
  input_float("start_age",&(E->control.start_age),"0.0",m);
  input_int("reset_startage",&(E->control.reset_startage),"0",m);
  input_int("zero_elapsed_time",&(E->control.zero_elapsed_time),"0",m);

  input_int("output_ll_max",&(E->output.llmax),"1",m);

  input_int("topvbc",&(E->mesh.topvbc),"0",m);
  input_int("botvbc",&(E->mesh.botvbc),"0",m);

  input_float("topvbxval",&(E->control.VBXtopval),"0.0",m);
  input_float("botvbxval",&(E->control.VBXbotval),"0.0",m);
  input_float("topvbyval",&(E->control.VBYtopval),"0.0",m);
  input_float("botvbyval",&(E->control.VBYbotval),"0.0",m);

  input_int("pseudo_free_surf",&(E->control.pseudo_free_surf),"0",m);

  input_int("toptbc",&(E->mesh.toptbc),"1",m);
  input_int("bottbc",&(E->mesh.bottbc),"1",m);
  input_float("toptbcval",&(E->control.TBCtopval),"0.0",m);
  input_float("bottbcval",&(E->control.TBCbotval),"1.0",m);

  input_boolean("side_sbcs",&(E->control.side_sbcs),"off",m);

  input_int("file_vbcs",&(E->control.vbcs_file),"0",m);
  input_string("vel_bound_file",E->control.velocity_boundary_file,"",m);

  input_int("mat_control",&(E->control.mat_control),"0",m);
  input_string("mat_file",E->control.mat_file,"",m);

  input_int("nodex",&(E->mesh.nox),"essential",m);
  input_int("nodez",&(E->mesh.noz),"essential",m);
  input_int("nodey",&(E->mesh.noy),"essential",m);

  input_boolean("aug_lagr",&(E->control.augmented_Lagr),"off",m);
  input_double("aug_number",&(E->control.augmented),"0.0",m);

  input_float("tole_compressibility",&(E->control.tole_comp),"0.0",m);

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

  input_int("vhighstep",&(E->control.v_steps_high),"1,0,nomax",m);
  input_int("vlowstep",&(E->control.v_steps_low),"250,0,nomax",m);
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
          input_float("relative_err_accuracy",&(E->control.relative_err_accuracy),"0.001",m);
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
    int m,n,snel,nsf,elx,ely,nox,noy,noz,nno,nel,npno;
    int k,i,j,d,l,nno_l,npno_l,nozl,nnov_l,nxyz;

    m=0;
    n=1;

 for (j=1;j<=E->sphere.caps_per_proc;j++)  {

  npno = E->lmesh.npno;
  nel  = E->lmesh.nel;
  nno  = E->lmesh.nno;
  nsf  = E->lmesh.nsf;
  noz  = E->lmesh.noz;
  nox  = E->lmesh.nox;
  noy  = E->lmesh.noy;
  elx  = E->lmesh.elx;
  ely  = E->lmesh.ely;

  E->P[j]        = (double *) malloc((npno+1)*sizeof(double));
  E->T[j]        = (double *) malloc((nno+1)*sizeof(double));
  E->NP[j]       = (float *) malloc((nno+1)*sizeof(float));
  E->edot[j]     = (float *) malloc((nno+1)*sizeof(float));
  E->buoyancy[j] = (double *) malloc((nno+1)*sizeof(double));

  E->gstress[j] = (float *) malloc((6*nno+1)*sizeof(float));
  E->stress[j]   = (float *) malloc((12*nsf+1)*sizeof(float));

  for(i=1;i<=E->mesh.nsd;i++)
      E->sphere.cap[j].TB[i] = (float *)  malloc((nno+1)*sizeof(float));

  E->age[j]      = (float *)malloc((nsf+2)*sizeof(float));

  E->slice.tpg[j]      = (float *)malloc((nsf+2)*sizeof(float));
  E->slice.tpgb[j]     = (float *)malloc((nsf+2)*sizeof(float));
  E->slice.divg[j]     = (float *)malloc((nsf+2)*sizeof(float));
  E->slice.vort[j]     = (float *)malloc((nsf+2)*sizeof(float));
  E->slice.shflux[j]    = (float *)malloc((nsf+2)*sizeof(float));
  E->slice.bhflux[j]    = (float *)malloc((nsf+2)*sizeof(float));
  /*  if(E->mesh.topvbc==2 && E->control.pseudo_free_surf) */
  E->slice.freesurf[j]    = (float *)malloc((nsf+2)*sizeof(float));

  E->mat[j] = (int *) malloc((nel+2)*sizeof(int));
  E->VIP[j] = (float *) malloc((nel+2)*sizeof(float));

  E->heating_adi[j]    = (double *) malloc((nel+1)*sizeof(double));
  E->heating_visc[j]   = (double *) malloc((nel+1)*sizeof(double));
  E->heating_latent[j] = (double *) malloc((nel+1)*sizeof(double));

  /* lump mass matrix for the energy eqn */
  E->TMass[j] = (double *) malloc((nno+1)*sizeof(double));

  nxyz = max(nox*noz,nox*noy);
  nxyz = 2*max(nxyz,noz*noy);

  E->sien[j]         = (struct SIEN *) malloc((nxyz+2)*sizeof(struct SIEN));
  E->surf_element[j] = (int *) malloc((nxyz+2)*sizeof(int));
  E->surf_node[j]    = (int *) malloc((nsf+2)*sizeof(int));

  }         /* end for cap j  */

  /* density field */
  E->rho      = (double *) malloc((nno+1)*sizeof(double));

  /* horizontal average */
  E->Have.T         = (float *)malloc((E->lmesh.noz+2)*sizeof(float));
  E->Have.V[1]      = (float *)malloc((E->lmesh.noz+2)*sizeof(float));
  E->Have.V[2]      = (float *)malloc((E->lmesh.noz+2)*sizeof(float));

 for(i=E->mesh.levmin;i<=E->mesh.levmax;i++) {
  E->sphere.R[i] = (double *)  malloc((E->lmesh.NOZ[i]+1)*sizeof(double));
  for (j=1;j<=E->sphere.caps_per_proc;j++)  {
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
      E->X[i][j][d]  = (double *)  malloc((nno+1)*sizeof(double));
      E->SX[i][j][d]  = (double *)  malloc((nno+1)*sizeof(double));
      }

    for(d=0;d<=3;d++)
      E->SinCos[i][j][d]  = (double *)  malloc((nno+1)*sizeof(double));

    E->IEN[i][j] = (struct IEN *)   malloc((nel+2)*sizeof(struct IEN));
    E->EL[i][j]  = (struct SUBEL *) malloc((nel+2)*sizeof(struct SUBEL));
    E->sphere.area1[i][j] = (double *) malloc((snel+1)*sizeof(double));
    for (k=1;k<=4;k++)
      E->sphere.angle1[i][j][k] = (double *) malloc((snel+1)*sizeof(double));

    E->MASS[i][j]     = (float *) malloc((nno+1)*sizeof(float));
    E->ECO[i][j] = (struct COORD *) malloc((nno+2)*sizeof(struct COORD));

    E->TWW[i][j] = (struct FNODE *)   malloc((nel+2)*sizeof(struct FNODE));

    for(d=1;d<=E->mesh.nsd;d++)
      for(l=1;l<=E->lmesh.NNO[i];l++)  {
        E->SX[i][j][d][l] = 0.0;
        E->X[i][j][d][l] = 0.0;
        }

    }
  }

 for(i=0;i<=E->output.llmax;i++)
  E->sphere.hindex[i] = (int *) malloc((E->output.llmax+3)
				       *sizeof(int));


 for(i=E->mesh.gridmin;i<=E->mesh.gridmax;i++)
  for (j=1;j<=E->sphere.caps_per_proc;j++)  {

    nno  = E->lmesh.NNO[i];
    npno = E->lmesh.NPNO[i];
    nel  = E->lmesh.NEL[i];
    nox = E->lmesh.NOX[i];
    noz = E->lmesh.NOZ[i];
    noy = E->lmesh.NOY[i];
    elx = E->lmesh.ELX[i];
    ely = E->lmesh.ELY[i];

    nxyz = elx*ely;
    E->CC[i][j] =(struct CC *)  malloc((1)*sizeof(struct CC));
    E->CCX[i][j]=(struct CCX *)  malloc((1)*sizeof(struct CCX));
    /* Test */
    E->ELEMENT[i][j] = (unsigned int *) malloc ((nel+2)*sizeof(unsigned int));

    for (k=1;k<=nel;k++)
       E->ELEMENT[i][j][k] = 0;
    /*ccccc*/

    E->elt_del[i][j] = (struct EG *) malloc((nel+1)*sizeof(struct EG));

    if(E->control.inv_gruneisen != 0)
        E->elt_c[i][j] = (struct EC *) malloc((nel+1)*sizeof(struct EC));

    E->EVI[i][j] = (float *) malloc((nel+2)*vpoints[E->mesh.nsd]*sizeof(float));
    E->BPI[i][j] = (double *) malloc((npno+1)*sizeof(double));

    E->ID[i][j]  = (struct ID *)    malloc((nno+2)*sizeof(struct ID));
    E->VI[i][j]  = (float *)        malloc((nno+2)*sizeof(float));
    E->NODE[i][j] = (unsigned int *)malloc((nno+2)*sizeof(unsigned int));

    nxyz = max(nox*noz,nox*noy);
    nxyz = 2*max(nxyz,noz*noy);
    nozl = max(noy,nox*2);



    E->parallel.EXCHANGE_sNODE[i][j] = (struct PASS *) malloc((nozl+2)*sizeof(struct PASS));
    E->parallel.NODE[i][j]   = (struct BOUND *) malloc((nxyz+2)*sizeof(struct BOUND));
    E->parallel.EXCHANGE_NODE[i][j]= (struct PASS *) malloc((nxyz+2)*sizeof(struct PASS));
    E->parallel.EXCHANGE_ID[i][j] = (struct PASS *) malloc((nxyz*E->mesh.nsd+3)*sizeof(struct PASS));

    for(l=1;l<=E->lmesh.NNO[i];l++)  {
      E->NODE[i][j][l] = (INTX | INTY | INTZ);  /* and any others ... */
      E->VI[i][j][l] = 1.0;
      }


    }         /* end for cap and i & j  */


 for (j=1;j<=E->sphere.caps_per_proc;j++)  {

  for(k=1;k<=E->mesh.nsd;k++)
    for(i=1;i<=E->lmesh.nno;i++)
      E->sphere.cap[j].TB[k][i] = 0.0;

  for(i=1;i<=E->lmesh.nno;i++)
     E->T[j][i] = 0.0;

  for(i=1;i<=E->lmesh.nel;i++)   {
      E->mat[j][i]=1;
      E->VIP[j][i]=1.0;

      E->heating_adi[j][i] = 0;
      E->heating_visc[j][i] = 0;
      E->heating_latent[j][i] = 1.0;
  }

  for(i=1;i<=E->lmesh.npno;i++)
      E->P[j][i] = 0.0;

  mat_prop_allocate(E);
  phase_change_allocate(E);
  set_up_nonmg_aliases(E,j);

  }         /* end for cap j  */

  if (strcmp(E->output.format, "hdf5") == 0)
      h5output_allocate_memory(E);

  return;
  }

/*  =========================================================  */

void allocate_velocity_vars(E)
     struct All_variables *E;

{
    int m,n,i,j,k,l;

 m=0;
 n=1;
  for (j=1;j<=E->sphere.caps_per_proc;j++)   {
    E->lmesh.nnov = E->lmesh.nno;
    E->lmesh.neq = E->lmesh.nnov * E->mesh.nsd;

    E->temp[j] = (double *) malloc((E->lmesh.neq+1)*sizeof(double));
    E->temp1[j] = (double *) malloc(E->lmesh.neq*sizeof(double));
    E->F[j] = (double *) malloc(E->lmesh.neq*sizeof(double));
    E->U[j] = (double *) malloc((E->lmesh.neq+1)*sizeof(double));
    E->u1[j] = (double *) malloc((E->lmesh.neq+1)*sizeof(double));


    for(i=1;i<=E->mesh.nsd;i++) {
      E->sphere.cap[j].V[i] = (float *) malloc((E->lmesh.nnov+1)*sizeof(float));
      E->sphere.cap[j].VB[i] = (float *)malloc((E->lmesh.nnov+1)*sizeof(float));
      E->sphere.cap[j].Vprev[i] = (float *) malloc((E->lmesh.nnov+1)*sizeof(float));
    }

    for(i=0;i<E->lmesh.neq;i++)
      E->U[j][i] = E->temp[j][i] = E->temp1[j][i] = 0.0;

    if(E->control.tracer == 1)  {
      for(i=1;i<=E->mesh.nsd;i++)     {
        E->GV[j][i]=(float*) malloc(((E->lmesh.nno+1)*E->parallel.nproc+1)*sizeof(float));
        E->GV1[j][i]=(float*) malloc(((E->lmesh.nno+1)*E->parallel.nproc+1)*sizeof(float));
        E->V[j][i]=(float*) malloc((E->lmesh.nno+1)*sizeof(float));

        for(k=0;k<(E->lmesh.nno+1)*E->parallel.nproc;k++)   {
          E->GV[j][i][k]=0.0;
          E->GV1[j][i][k]=0.0;
        }
      }
    }

    for(k=1;k<=E->mesh.nsd;k++)
      for(i=1;i<=E->lmesh.nnov;i++)
        E->sphere.cap[j].VB[k][i] = 0.0;

  }       /* end for cap j */

  for(l=E->mesh.gridmin;l<=E->mesh.gridmax;l++)
    for (j=1;j<=E->sphere.caps_per_proc;j++)   {
      E->lmesh.NEQ[l] = E->lmesh.NNOV[l] * E->mesh.nsd;

      E->BI[l][j] = (double *) malloc((E->lmesh.NEQ[l])*sizeof(double));
      k = (E->lmesh.NOX[l]*E->lmesh.NOZ[l]+E->lmesh.NOX[l]*E->lmesh.NOY[l]+
          E->lmesh.NOY[l]*E->lmesh.NOZ[l])*6;
      E->zero_resid[l][j] = (int *) malloc((k+2)*sizeof(int));
      E->parallel.Skip_id[l][j] = (int *) malloc((k+2)*sizeof(int));

      for(i=0;i<E->lmesh.NEQ[l];i++) {
         E->BI[l][j][i]=0.0;
         }

      }   /* end for j & l */

  return;
 }


/*  =========================================================  */

void global_default_values(E)
     struct All_variables *E;
{

  /* FIRST: values which are not changed routinely by the user */

  E->control.v_steps_low = 10;
  E->control.v_steps_upper = 1;
  E->control.accuracy = 1.0e-6;
  E->control.vaccuracy = 1.0e-8;
  E->control.verbose=0; /* debugging/profiles */

  /* SECOND: values for which an obvious default setting is useful */

  E->control.CITCOM_ORTHO = 1; /* for orthogonal meshes by default */
  E->control.CITCOM_ORTHOZ = 1; /* for orthogonal meshes by default */


    E->control.stokes=0;
    E->control.restart=0;
    E->control.CONVECTION = 0;
    E->control.CART2D = 0;
    E->control.CART3D = 0;
    E->control.CART2pt5D = 0;
    E->control.AXI = 0;
    E->control.CONJ_GRAD = 0;
    E->control.NMULTIGRID = 0;
    E->control.EMULTIGRID = 0;
    E->control.COMPRESS = 1;
    E->control.augmented_Lagr = 0;
    E->control.augmented = 0.0;

    E->control.GRID_TYPE=1;

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
  E->data.grav_const = 6.673e-11;
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

  return;  }


/* =============================================================
   ============================================================= */

void check_bc_consistency(E)
     struct All_variables *E;

{ int i,j,lev;

  for (j=1;j<=E->sphere.caps_per_proc;j++)  {
    for(i=1;i<=E->lmesh.nno;i++)    {
      if ((E->node[j][i] & VBX) && (E->node[j][i] & SBX))
        printf("Inconsistent x velocity bc at %d\n",i);
      if ((E->node[j][i] & VBZ) && (E->node[j][i] & SBZ))
        printf("Inconsistent z velocity bc at %d\n",i);
      if ((E->node[j][i] & VBY) && (E->node[j][i] & SBY))
        printf("Inconsistent y velocity bc at %d\n",i);
      if ((E->node[j][i] & TBX) && (E->node[j][i] & FBX))
        printf("Inconsistent x temperature bc at %d\n",i);
      if ((E->node[j][i] & TBZ) && (E->node[j][i] & FBZ))
        printf("Inconsistent z temperature bc at %d\n",i);
      if ((E->node[j][i] & TBY) && (E->node[j][i] & FBY))
        printf("Inconsistent y temperature bc at %d\n",i);
      }
    }          /* end for j */

  for(lev=E->mesh.gridmin;lev<=E->mesh.gridmax;lev++)
    for (j=1;j<=E->sphere.caps_per_proc;j++)  {
      for(i=1;i<=E->lmesh.NNO[lev];i++)        {
        if ((E->NODE[lev][j][i] & VBX) && (E->NODE[lev][j][i]  & SBX))
          printf("Inconsistent x velocity bc at %d,%d\n",lev,i);
        if ((E->NODE[lev][j][i] & VBZ) && (E->NODE[lev][j][i]  & SBZ))
          printf("Inconsistent z velocity bc at %d,%d\n",lev,i);
        if ((E->NODE[lev][j][i] & VBY) && (E->NODE[lev][j][i]  & SBY))
          printf("Inconsistent y velocity bc at %d,%d\n",lev,i);
        /* Tbc's not applicable below top level */
        }

    }   /* end for  j and lev */

  return;

}

void set_up_nonmg_aliases(E,j)
     struct All_variables *E;
     int j;

{ /* Aliases for functions only interested in the highest mg level */

  int i;

  E->eco[j] = E->ECO[E->mesh.levmax][j];
  E->ien[j] = E->IEN[E->mesh.levmax][j];
  E->id[j] = E->ID[E->mesh.levmax][j];
  E->Vi[j] = E->VI[E->mesh.levmax][j];
  E->EVi[j] = E->EVI[E->mesh.levmax][j];
  E->node[j] = E->NODE[E->mesh.levmax][j];
  E->cc[j] = E->CC[E->mesh.levmax][j];
  E->ccx[j] = E->CCX[E->mesh.levmax][j];
  E->Mass[j] = E->MASS[E->mesh.levmax][j];
  E->element[j] = E->ELEMENT[E->mesh.levmax][j];

  for (i=1;i<=E->mesh.nsd;i++)    {
    E->x[j][i] = E->X[E->mesh.levmax][j][i];
    E->sx[j][i] = E->SX[E->mesh.levmax][j][i];
    }

  return; }

void report(E,string)
     struct All_variables *E;
     char * string;
{ if(E->control.verbose && E->parallel.me==0)
    { fprintf(stderr,"%s\n",string);
      fflush(stderr);
    }
  return;
}

void record(E,string)
     struct All_variables *E;
     char * string;
{ if(E->control.verbose)
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
    /*void read_viscosity_option();*/
    void initial_viscosity();

    report(E,"Initialize pressure field");
    initial_pressure(E);
    report(E,"Initialize velocity field");
    initial_velocity(E);
    report(E,"Initialize viscosity field");
    /*get_viscosity_option(E);*/
    initial_viscosity(E);

    return;

}

/* ========================================== */

void initial_pressure(E)
     struct All_variables *E;
{
    int i,m;

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=1;i<=E->lmesh.npno;i++)
      E->P[m][i]=0.0;

  return;
}

void initial_velocity(E)
     struct All_variables *E;
{
    int i,m;

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=1;i<=E->lmesh.nnov;i++)   {
        E->sphere.cap[m].V[1][i]=0.0;
        E->sphere.cap[m].V[2][i]=0.0;
        E->sphere.cap[m].V[3][i]=0.0;
        }

    return;
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
      E->convection.tic_method == -1 ||
      (E->control.tracer && E->trace.ic_method == 2)) {
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
        E->convection.tic_method == -1 ||
        (E->control.tracer && E->trace.ic_method == 2)) {
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
#ifdef USE_GZDIR
    else if (strcmp(E->output.format, "ascii-gz") == 0)
        E->problem_output = gzdir_output;
    else {
        /* indicate error here */
        if (E->parallel.me == 0) {
            fprintf(stderr, "wrong output_format, must be 'ascii', 'hdf5', or 'ascii-gz'\n");
            fprintf(E->fp, "wrong output_format, must be  'ascii', 'hdf5', or 'ascii-gz'\n");
        }
        parallel_process_termination(E);
    }
#else
    else {
        /* indicate error here */
        if (E->parallel.me == 0) {
            fprintf(stderr, "wrong output_format, must be 'ascii' or 'hdf5' (USE_GZDIR undefined)\n");
            fprintf(E->fp, "wrong output_format, must be 'ascii' or 'hdf5' (USE_GZDIR undefined)\n");
        }
        parallel_process_termination(E);
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

  if(E->output.fpqt)
    fclose(E->output.fpqt);
  if(E->output.fpqb)
    fclose(E->output.fpqb);


#ifdef USE_GZDIR
  /*
     remove VTK geo file in case we used that for IO
  */
  if((E->output.gzdir.vtk_io != 1) &&
     (strcmp(E->output.format, "ascii-gz") == 0)){
    if((E->output.gzdir.vtk_io == 3)||(E->parallel.me == 0)){
      /* delete the geo files */
      get_vtk_filename(files,1,E,0);
      sprintf(message,"rm %s",files);system(message);
      if(E->parallel.me == 0){
	/* close the log */
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


