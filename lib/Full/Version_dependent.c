
#include <signal.h>
#include <math.h>
#include <sys/types.h>
#include <string.h>
#include "element_definitions.h"
#include "global_defs.h"

// Setup global mesh parameters
//
void global_derived_values(E)
     struct All_variables *E;
{
    int d,lx,lz,ly,i,nox,noz,noy;
    char logfile[100], timeoutput[100];
    FILE *fp, *fptime;
    void parallel_process_termination();
 
    /* As early as possible, set up the log file to 
       record information about the progress of the 
       program as it runs 
       */
    /* also add a time file to output time CPC 6/18/00 */

    sprintf(logfile,"%s.log",E->control.data_file);
    sprintf(timeoutput,"%s.time",E->control.data_file);

    if((fp=fopen(logfile,"w")) == NULL)
	E->fp = stdout;
    else
	E->fp = fp;

    if((fptime=fopen(timeoutput,"w")) == NULL)
	E->fptime = stdout;
    else
	E->fptime = fptime;

   E->mesh.levmax = E->mesh.levels-1;
   nox = E->mesh.mgunitx * (int) pow(2.0,((double)E->mesh.levmax))*E->parallel.nprocx + 1;
   noy = E->mesh.mgunity * (int) pow(2.0,((double)E->mesh.levmax))*E->parallel.nprocy + 1; 
   noz = E->mesh.mgunitz * (int) pow(2.0,((double)E->mesh.levmax))*E->parallel.nprocz + 1; 

   if (E->control.NMULTIGRID||E->control.EMULTIGRID)  {
      E->mesh.levmax = E->mesh.levels-1;
      E->mesh.gridmax = E->mesh.levmax; 
      E->mesh.nox = E->mesh.mgunitx * (int) pow(2.0,((double)E->mesh.levmax))*E->parallel.nprocx + 1;
      E->mesh.noy = E->mesh.mgunity * (int) pow(2.0,((double)E->mesh.levmax))*E->parallel.nprocy + 1; 
      E->mesh.noz = E->mesh.mgunitz * (int) pow(2.0,((double)E->mesh.levmax))*E->parallel.nprocz + 1; 
      }
   else   {
      if (nox!=E->mesh.nox || noy!=E->mesh.noy || noz!=E->mesh.noz) {
         if (E->parallel.me==0) 
            fprintf(stderr,"inconsistent mesh for interpolation, quit the run\n");
         parallel_process_termination();
         }
      E->mesh.gridmax = E->mesh.levmax; 
      E->mesh.gridmin = E->mesh.levmax; 
     }

   if(E->mesh.nsd != 3) 
      E->mesh.noy = 1;

   E->mesh.nnx[1] = E->mesh.nox;	
   E->mesh.nnx[2] = E->mesh.noy;	
   E->mesh.nnx[3] = E->mesh.noz;	
   E->mesh.elx = E->mesh.nox-1;	
   E->mesh.ely = E->mesh.noy-1;
   E->mesh.elz = E->mesh.noz-1;

   E->mesh.nno = E->sphere.caps;
   for(d=1;d<=E->mesh.nsd;d++) 
      E->mesh.nno *= E->mesh.nnx[d];
       
   E->mesh.nel = E->sphere.caps*E->mesh.elx*E->mesh.elz*E->mesh.ely;

   E->mesh.nnov = E->mesh.nno;

   E->mesh.neq = E->mesh.nnov*E->mesh.nsd;

   E->mesh.npno = E->mesh.nel;
   E->mesh.nsf = E->mesh.nox*E->mesh.noy;

   for(i=E->mesh.levmax;i>=E->mesh.levmin;i--) {
      if (E->control.NMULTIGRID||E->control.EMULTIGRID)
	{ nox = E->mesh.mgunitx * (int) pow(2.0,(double)i) + 1;
	  noy = E->mesh.mgunity * (int) pow(2.0,(double)i) + 1;
	  noz = E->mesh.mgunitz * (int) pow(2.0,(double)i) + 1;
	}
      else 
	{ noz = E->mesh.noz;
	  nox = E->mesh.nox;
	  noy = E->mesh.noy;
          /*if (i<E->mesh.levmax) noz=2;*/
	}

      E->mesh.ELX[i] = nox-1;
      E->mesh.ELY[i] = noy-1;
      E->mesh.ELZ[i] = noz-1;
      E->mesh.NNO[i] = E->sphere.caps * nox * noz * noy;
      E->mesh.NEL[i] = E->sphere.caps * (nox-1) * (noz-1) * (noy-1);
      E->mesh.NPNO[i] = E->mesh.NEL[i] ;
      E->mesh.NOX[i] = nox;
      E->mesh.NOZ[i] = noz;
      E->mesh.NOY[i] = noy;

      E->mesh.NNOV[i] = E->mesh.NNO[i];
      E->mesh.NEQ[i] = E->mesh.nsd * E->mesh.NNOV[i] ;  

      }

    E->sphere.elx = E->sphere.nox-1;
    E->sphere.ely = E->sphere.noy-1;
    E->sphere.snel = E->sphere.ely*E->sphere.elx;
    E->sphere.nsf = E->sphere.noy*E->sphere.nox;

    E->data.scalet = (E->data.layer_km*E->data.layer_km/E->data.therm_diff)/(1.e6*365.25*24*3600);
    E->data.scalev = (E->data.layer_km/E->data.therm_diff)/(100.0*365.25*24*3600);
    E->data.timedir = E->control.Atemp / fabs(E->control.Atemp);

    if(E->control.print_convergence && E->parallel.me==0)
	fprintf(stderr,"Problem has %d x %d x %d nodes\n",E->mesh.nox,E->mesh.noz,E->mesh.noy);

   return; 
} 


void read_initial_settings(E)
     struct All_variables *E;
  
{
    void set_convection_defaults();
    void set_2dc_defaults();
    void set_3dc_defaults();
    void set_3dsphere_defaults();
    void set_cg_defaults();
    void set_mg_defaults();
    int m=E->parallel.me;

  /* first the problem type (defines subsequent behaviour) */

    input_string("Problem",E->control.PROBLEM_TYPE,NULL,m);
    if ( strcmp(E->control.PROBLEM_TYPE,"convection") == 0)  {
	E->control.CONVECTION = 1; 
	set_convection_defaults(E);
    }

    else if ( strcmp(E->control.PROBLEM_TYPE,"convection-chemical") == 0) {
	E->control.CONVECTION = 1;
	E->control.CHEMISTRY_MODULE=1;
	set_convection_defaults(E);
    }
    
    else {
	fprintf(E->fp,"Unable to determine problem type, assuming convection ... \n");
	E->control.CONVECTION = 1;
	set_convection_defaults(E);
    }
      
  input_string("Geometry",E->control.GEOMETRY,NULL,m); 
  if ( strcmp(E->control.GEOMETRY,"cart2d") == 0)
    { E->control.CART2D = 1; 
      set_2dc_defaults(E);}
  else if ( strcmp(E->control.GEOMETRY,"axi") == 0)
    { E->control.AXI = 1; 
      }
  else if ( strcmp(E->control.GEOMETRY,"cart2pt5d") == 0)
    { E->control.CART2pt5D = 1; 
      set_2pt5dc_defaults(E);}
  else if ( strcmp(E->control.GEOMETRY,"cart3d") == 0)
    { E->control.CART3D = 1;
      set_3dc_defaults(E);}
  else if ( strcmp(E->control.GEOMETRY,"sphere") == 0)
    { 
      set_3dsphere_defaults(E);}
  else
    { fprintf(E->fp,"Unable to determine geometry, assuming cartesian 2d ... \n");
      E->control.CART2D = 1; 
      set_2dc_defaults(E); }

  input_string("Solver",E->control.SOLVER_TYPE,NULL,m);
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
      exit(0); 
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
    
/*     input_string("datatypes",E->control.which_data_files,"",m); */
/*     input_string("averages",E->control.which_horiz_averages,"",m); */
/*     input_string("timelog",E->control.which_running_data,"",m); */
/*     input_string("observables",E->control.which_observable_data,"",m); */

    input_string("datafile",E->control.data_file,"initialize",m);
    input_string("datafile_old",E->control.old_P_file,"initialize",m);

/*     input_string("process_command",E->control.output_written_external_command,"",m); */
/*     input_boolean("AVS",&(E->control.AVS),"off",m); */
/*     input_boolean("CONMAN",&(E->control.CONMAN),"off",m); */
/*     input_boolean("modify_slab",&(E->control.SLAB),"off",m); */
    
 /*   if (E->control.NMULTIGRID||E->control.EMULTIGRID) {
	input_int("mgunitx",&(E->mesh.mgunitx),"1",m);
	input_int("mgunitz",&(E->mesh.mgunitz),"1",m);
	input_int("mgunity",&(E->mesh.mgunity),"1",m);
	input_int("levels",&(E->mesh.levels),"0",m);
    }
*/
	input_int("mgunitx",&(E->mesh.mgunitx),"1",m);
	input_int("mgunitz",&(E->mesh.mgunitz),"1",m);
	input_int("mgunity",&(E->mesh.mgunity),"1",m);
	input_int("levels",&(E->mesh.levels),"0",m);

        input_int("nprocx",&(E->parallel.nprocx),"1",m);
        input_int("nprocy",&(E->parallel.nprocy),"1",m);
        input_int("nprocz",&(E->parallel.nprocz),"1",m);
        input_int("nproc_surf",&(E->parallel.nprocxy),"1",m);

/* 	input_int("relaxation",&(E->control.dfact),"0",m); */

    input_boolean("node_assemble",&(E->control.NASSEMBLE),"off",m);
                                    /* general mesh structure */

    input_boolean("verbose",&(E->control.verbose),"off",m);
    input_boolean("see_convergence",&(E->control.print_convergence),"off",m);
/*     input_boolean("COMPRESS",&(E->control.COMPRESS),"on",m); */
/*     input_float("sobtol",&(E->control.sob_tolerance),"0.0001",m); */

    input_int("stokes_flow_only",&(E->control.stokes),"0",m);

    input_int("tracer",&(E->control.tracer),"0");
    input_string("tracer_file",E->control.tracer_file," ");

    input_int("restart",&(E->control.restart),"0",m);
    input_int("post_p",&(E->control.post_p),"0",m);
    input_int("solution_cycles_init",&(E->monitor.solution_cycles_init),"0",m);

        /* for layers    */
/*     input_int("nz_lmantle",&(E->viscosity.nlm),"1",m); */
/*     input_int("nz_410",&(E->viscosity.n410),"1",m); */
/*     input_int("nz_lith",&(E->viscosity.nlith),"1",m); */
    input_float("z_cmb",&(E->viscosity.zcmb),"1.0",m);
    input_float("z_lmantle",&(E->viscosity.zlm),"1.0",m);
    input_float("z_410",&(E->viscosity.z410),"1.0",m);
    input_float("z_lith",&(E->viscosity.zlith),"0.0",m);

/*  the start age and initial subduction history   */
    input_float("start_age",&(E->control.start_age),"0.0",m);
    input_int("reset_startage",&(E->control.reset_startage),"0",m);
    input_int("zero_elapsed_time",&(E->control.zero_elapsed_time),"0",m);

    input_int("ll_max",&(E->sphere.llmax),"1",m);
    input_int("nlong",&(E->sphere.noy),"1",m);
    input_int("nlati",&(E->sphere.nox),"1",m);

    input_int("output_ll_max",&(E->sphere.output_llmax),"1",m);
    input_int("slab_model_layers",&(E->sphere.slab_layers),"1",m);

/*     input_int("read_slab",&(E->control.read_slab),"1",m); */
/*     input_int("read_density",&(E->control.read_density),"1",m); */
/*     if (E->control.read_density && E->control.read_slab) */
/*         E->control.read_slab = 0; */

        /* for phase change    */
    input_float("Ra_410",&(E->control.Ra_410),"0.0",m);
    input_float("clapeyron410",&(E->control.clapeyron410),"0.0",m);
    input_float("transT410",&(E->control.transT410),"0.0",m);
    input_float("width410",&(E->control.width410),"0.0",m);

    if (E->control.width410!=0.0)
       E->control.width410 = 1.0/E->control.width410;

    input_float("Ra_670",&(E->control.Ra_670),"0.0",m);
    input_float("clapeyron670",&(E->control.clapeyron670),"0.0",m);
    input_float("transT670",&(E->control.transT670),"0.0",m);
    input_float("width670",&(E->control.width670),"0.0",m);

    if (E->control.width670!=0.0)
       E->control.width670 = 1.0/E->control.width670;
    
    input_float("Ra_cmb",&(E->control.Ra_cmb),"0.0",m);
    input_float("clapeyroncmb",&(E->control.clapeyroncmb),"0.0",m);
    input_float("transTcmb",&(E->control.transTcmb),"0.0",m);
    input_float("widthcmb",&(E->control.widthcmb),"0.0",m);

    if (E->control.widthcmb!=0.0)
       E->control.widthcmb = 1.0/E->control.widthcmb;

    input_int("topvbc",&(E->mesh.topvbc),"0",m);
    input_int("botvbc",&(E->mesh.botvbc),"0",m);
/*     input_int("sidevbc",&(E->mesh.sidevbc),"0",m); */
/*     >>>>Tan2 1/22/02 for time-varied imposed vbc */
    input_int("file_vbcs",&(E->control.vbcs_file),"0",m);
    input_string("vel_bound_file",E->control.velocity_boundary_file,"",m);

    input_int("mat_control",&(E->control.mat_control),"0",m);
    input_string("mat_file",E->control.mat_file,"",m);

/*     input_boolean("periodicx",&(E->mesh.periodic_x),"off",m); */
/*     input_boolean("periodicy",&(E->mesh.periodic_y),"off",m); */
/*     input_boolean("depthdominated",&(E->control.depth_dominated),"off",m); */
/*     input_boolean("eqnzigzag",&(E->control.eqn_zigzag),"off",m); */
/*     input_boolean("eqnviscosity",&(E->control.eqn_viscosity),"off",m); */

    input_float("topvbxval",&(E->control.VBXtopval),"0.0",m);
    input_float("botvbxval",&(E->control.VBXbotval),"0.0",m);
    input_float("topvbyval",&(E->control.VBYtopval),"0.0",m);
    input_float("botvbyval",&(E->control.VBYbotval),"0.0",m);
  
    input_int("toptbc",&(E->mesh.toptbc),"1",m);
    input_int("bottbc",&(E->mesh.bottbc),"1",m);
    input_float("toptbcval",&(E->control.TBCtopval),"0.0",m);
    input_float("bottbcval",&(E->control.TBCbotval),"1.0",m);
 
/*     input_float("blyr_hwx1",&(E->mesh.bl1width[1]),"0.0",m); */
/*     input_float("blyr_hwz1",&(E->mesh.bl1width[2]),"0.0",m); */
/*     input_float("blyr_hwy1",&(E->mesh.bl1width[3]),"0.0",m); */
/*     input_float("blyr_hwx2",&(E->mesh.bl2width[1]),"0.0",m); */
/*     input_float("blyr_hwz2",&(E->mesh.bl2width[2]),"0.0",m); */
/*     input_float("blyr_hwy2",&(E->mesh.bl2width[3]),"0.0",m); */
/*     input_float("blyr_mgx1",&(E->mesh.bl1mag[1]),"0.0",m); */
/*     input_float("blyr_mgz1",&(E->mesh.bl1mag[2]),"0.0",m); */
/*     input_float("blyr_mgy1",&(E->mesh.bl1mag[3]),"0.0",m); */
/*     input_float("blyr_mgx2",&(E->mesh.bl2mag[1]),"0.0",m); */
/*     input_float("blyr_mgz2",&(E->mesh.bl2mag[2]),"0.0",m); */
/*     input_float("blyr_mgy2",&(E->mesh.bl2mag[3]),"0.0",m); */
   
   
/*     input_float("region_wdx",&(E->mesh.width[1]),"0.0",m); */
/*     input_float("region_wdz",&(E->mesh.width[2]),"0.0",m); */
/*     input_float("region_wdy",&(E->mesh.width[3]),"0.0",m); */
/*     input_float("region_hwx",&(E->mesh.hwidth[1]),"0.0",m); */
/*     input_float("region_hwz",&(E->mesh.hwidth[2]),"0.0",m); */
/*     input_float("region_hwy",&(E->mesh.hwidth[3]),"0.0",m); */
/*     input_float("region_mgx",&(E->mesh.magnitude[1]),"0.0",m); */
/*     input_float("region_mgz",&(E->mesh.magnitude[2]),"0.0",m); */
/*     input_float("region_mgy",&(E->mesh.magnitude[3]),"0.0",m); */
/*     input_float("region_ofx",&(E->mesh.offset[1]),"0.0",m); */
/*     input_float("region_ofz",&(E->mesh.offset[2]),"0.0",m); */
/*     input_float("region_ofy",&(E->mesh.offset[3]),"0.0",m); */

/*     input_string("gridxfile",E->mesh.gridfile[1],"",m); */
/*     input_string("gridzfile",E->mesh.gridfile[2],"",m); */
/*     input_string("gridyfile",E->mesh.gridfile[3],"",m); */
    
    
    input_float("dimenx",&(E->mesh.layer[1]),"1.0",m);
    input_float("dimenz",&(E->mesh.layer[2]),"1.0",m);
    input_float("dimeny",&(E->mesh.layer[3]),"1.0",m);
    

    input_int("nodex",&(E->mesh.nox),"essential",m);
    input_int("nodez",&(E->mesh.noz),"essential",m);
    input_int("nodey",&(E->mesh.noy),"essential",m);

    input_boolean("aug_lagr",&(E->control.augmented_Lagr),"off",m);
    input_double("aug_number",&(E->control.augmented),"0.0",m);
/*     input_float("jacobi_damping",&(E->control.jrelax),"1.0",m); */

    input_float("tole_compressibility",&(E->control.tole_comp),"0.0",m);
/*     input_boolean("orthogonal",&(E->control.ORTHO),"on",m); */
/*     input_boolean("crust",&(E->control.crust),"off",m); */
/*     input_float("crust_width",&(E->crust.width),"0.0",m); */

    input_int("storage_spacing",&(E->control.record_every),"10",m);
    input_int("cpu_limits_in_seconds",&(E->control.record_all_until),"5",m);
 
    input_boolean("precond",&(E->control.precondition),"off",m);
/*     input_boolean("vprecond",&(E->control.vprecondition),"on",m); */
    input_int("mg_cycle",&(E->control.mg_cycle),"2,0,nomax",m);
    input_int("down_heavy",&(E->control.down_heavy),"1,0,nomax",m);
    input_int("up_heavy",&(E->control.up_heavy),"1,0,nomax",m);
    input_double("accuracy",&(E->control.accuracy),"1.0e-4,0.0,1.0",m);
/*     input_int("viterations",&(E->control.max_vel_iterations),"250,0,nomax",m); */
/*      input_int("comm_line_min",&(E->control.comm_line_min),"1,0,nomax",m); */
 
    input_int("vhighstep",&(E->control.v_steps_high),"1,0,nomax",m);
    input_int("vlowstep",&(E->control.v_steps_low),"250,0,nomax",m);
/*      input_int("vupperstep",&(E->control.v_steps_upper),"1,0,nomax",m); */
    input_int("piterations",&(E->control.p_iterations),"100,0,nomax",m);
/*     input_int("maxsamevisc",&(E->control.max_same_visc),"25,0,nomax",m); */

  /* data section */ 

/*    input_float("ReferenceT",&(E->data.ref_temperature),"2600.0",m); */
  input_float("Q0",&(E->control.Q0),"0.0",m);
  input_float("layerd",&(E->data.layer_km),"2800.0",m);
  input_float("gravacc",&(E->data.grav_acc),"9.81",m);
  input_float("thermexp",&(E->data.therm_exp),"3.28e-5",m);
  input_float("cp",&(E->data.Cp),"1200.0",m);
  input_float("thermdiff",&(E->data.therm_diff),"8.0e-7",m);
/*    input_float("thermcond",&(E->data.therm_cond),"3.168",m); */
  input_float("density",&(E->data.density),"3340.0",m);
  input_float("wdensity",&(E->data.density_above),"1030.0",m);
/*   input_float("mdensity",&(E->data.melt_density),"2800.0",m); */
/*   input_float("rdensity",&(E->data.res_density),"3295.0",m); */
/*   input_float("heatflux",&(E->data.surf_heat_flux),"4.4e-2",m); */
  input_float("refvisc",&(E->data.ref_viscosity),"1.0e21",m);
/*   input_float("meltvisc",&(E->data.melt_viscosity),"1.0e18",m); */
/*   input_float("surftemp",&(E->data.surf_temp),"0.0",m); */
/*   input_float("youngs",&(E->data.youngs_mod),"1.0e11",m); */
/*   input_float("Te",&(E->data.Te),"0.0",m); */
/*   input_float("Tsol0",&(E->data.T_sol0),"1373.0",m); */
/*   input_float("dTsoldz",&(E->data.dTsol_dz),"3.4e-3",m); */
/*   input_float("dTsoldF",&(E->data.dTsol_dF),"440.0",m); */
/*   input_float("dTdz",&(E->data.dT_dz),"0.48e-3",m); */
/*   input_float("deltaS",&(E->data.delta_S),"250.0",m); */
/*   input_float("gasconst",&(E->data.gas_const),"8.3",m); */   /* not much cause to change these ! */
/*   input_float("gravconst",&(E->data.grav_const),"6.673e-11",m); */
/*   input_float("permeability",&(E->data.permeability),"3.0e-10",m); */

 (E->problem_settings)(E);


 return; 
}


/* =================================================
   Standard node positions including mesh refinement 

   =================================================  */

void node_locations(E)
     struct All_variables *E;
{ 
  int m,i,j,k,ii,d,node,lev;
  double ro,ri,dr,*rr,*RR,fo;
  float t1,f1,tt1;
  int noz,lnoz,step,nn;
  char output_file[255], a[255];
  FILE *fp1;

  const int dims = E->mesh.nsd;

  void coord_of_cap();
  void rotate_mesh ();
  void compute_angle_surf_area ();
  void parallel_process_termination();

  rr = (double *)  malloc((E->mesh.noz+1)*sizeof(double));
  RR = (double *)  malloc((E->mesh.noz+1)*sizeof(double));

  if(E->control.coor==1)    {
      sprintf(output_file,"%s",E->control.coor_file);
      fp1=fopen(output_file,"r");
	if (fp1 == NULL) {
          fprintf(E->fp,"(Nodal_mesh.c #1) Cannot open %s\n",output_file);
          exit(8);
	}
      fscanf(fp1,"%s%d",&a,&i);
      if (i != E->mesh.noz ) {
          fprintf(E->fp,"(Nodal_mesh.c #2) inconsistent file length: %s\n",output_file);
          exit(8);
      }
      for (k=1;k<=E->mesh.noz;k++)  { 
	fscanf(fp1,"%d %f",&nn,&tt1);
	rr[k]=tt1;
      }

      fclose(fp1);
  }
  else {
    /* generate uniform mesh in radial direction */
    dr = (E->sphere.ro-E->sphere.ri)/(E->mesh.noz-1);

    for (k=1;k<=E->mesh.noz;k++)  {
      rr[k] = E->sphere.ri + (k-1)*dr;
    }
  }

  for (i=1;i<=E->lmesh.noz;i++)  {
      k = E->lmesh.nzs+i-1;
      RR[i] = rr[k];
      }

  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++) {

    if (E->control.NMULTIGRID||E->control.EMULTIGRID)
        step = (int) pow(2.0,(double)(E->mesh.levmax-lev));
    else
        step = 1;

      for (i=1;i<=E->lmesh.NOZ[lev];i++)
         E->sphere.R[lev][i] = RR[(i-1)*step+1];

    }          /* lev   */

  free ((void *) rr);
  free ((void *) RR);

  ro = -0.5*(M_PI/4.0)/E->mesh.elx;
  fo = 0.0;

  E->sphere.dircos[1][1] = cos(ro)*cos(fo);
  E->sphere.dircos[1][2] = cos(ro)*sin(fo);
  E->sphere.dircos[1][3] = -sin(ro);
  E->sphere.dircos[2][1] = -sin(fo);
  E->sphere.dircos[2][2] = cos(fo);
  E->sphere.dircos[2][3] = 0.0;
  E->sphere.dircos[3][1] = sin(ro)*cos(fo);
  E->sphere.dircos[3][2] = sin(ro)*sin(fo);
  E->sphere.dircos[3][3] = cos(ro);

  for (j=1;j<=E->sphere.caps_per_proc;j++)   {
     ii = E->sphere.capid[j];
     coord_of_cap(E,j,ii);
     }  

  /* rotate the mesh to avoid two poles on mesh points */
  for (j=1;j<=E->sphere.caps_per_proc;j++)   {
     ii = E->sphere.capid[j];
     rotate_mesh(E,j,ii);
     }

  compute_angle_surf_area (E);   /* used for interpolation */

  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++) 
    for (j=1;j<=E->sphere.caps_per_proc;j++)  
      for (i=1;i<=E->lmesh.NNO[lev];i++)  {
        E->SinCos[lev][j][0][i] = sin(E->SX[lev][j][1][i]);
        E->SinCos[lev][j][1][i] = sin(E->SX[lev][j][2][i]);
        E->SinCos[lev][j][2][i] = cos(E->SX[lev][j][1][i]);
        E->SinCos[lev][j][3][i] = cos(E->SX[lev][j][2][i]);
        }

  /*
if (E->control.verbose)
  for (lev=E->mesh.levmin;lev<=E->mesh.levmax;lev++)   {
    fprintf(E->fp_out,"output_coordinates after rotation %d \n",lev);
    for (j=1;j<=E->sphere.caps_per_proc;j++)
      for (i=1;i<=E->lmesh.NNO[lev];i++)
        if(i%E->lmesh.NOZ[lev]==1)
             fprintf(E->fp_out,"%d %d %g %g %g\n",j,i,E->SX[lev][j][1][i],E->SX[lev][j][2][i],E->SX[lev][j][3][i]);
      }
  */



return;
 }

