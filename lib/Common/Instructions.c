/* Set up the finite element problem to suit: returns with all memory */
/* allocated, temperature, viscosity, node locations and how to use */
/* them all established. 8.29.92 or 29.8.92 depending on your nationality*/

#include <signal.h>
#include <math.h>
#include <sys/types.h>
#include <string.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "citcom_init.h"

int Emergency_stop;

void read_instructions(filename)
     char *filename;
{
    int get_process_identifier();

    void allocate_common_vars();  
    void common_initial_fields();
    void read_initial_settings();
    void tracer_initial_settings();
    void global_default_values();
    void global_derived_values();
    void construct_ien();
    void parallel_domain_boundary_nodes();
    void construct_masks();
    void construct_shape_functions();
    void construct_id();
    void construct_lm();
    void construct_sub_element();
    void mass_matrix();
    void construct_node_ks();
    void construct_node_maps();
    void read_mat_from_file();
    void construct_mat_group();
    void interuption();
    void set_up_nonmg_aliases();
    void check_bc_consistency();
    void node_locations();
    void parallel_domain_decomp0();
    void parallel_communication_routs_v();
    void parallel_communication_routs_s();
    void parallel_processor_setup();
    void allocate_velocity_vars();                 
    void construct_c3x3matrix();
    void construct_surf_det ();
    void set_sphere_harmonics ();

    void setup_parser();

    void get_initial_elapsed_time();
    void set_starting_age();
    void set_elapsed_time();   

    void parallel_process_termination();
    void parallel_process_sync();
  
  
    double start_time, CPU_time0(),vmag;
    double global_vdot();
    char output_file[255];

    int *temp, i;
 
    /* =====================================================
       Global interuption handling routine defined once here
       =====================================================  */

    if (E->parallel.me==0) start_time=CPU_time0();
    Emergency_stop = 0;

    signal(SIGINT,interuption);
    signal(SIGTERM,interuption);

    E->control.PID=get_process_identifier(); 

    /* ==================================================
       Initialize from the command line 
       from startup files. (See Parsing.c).
       ==================================================  */

    setup_parser(E,filename);

    global_default_values(E); 
    read_initial_settings(E); 
    tracer_initial_settings(E);
 
    (E->problem_derived_values)(E);   /* call this before global_derived_  */
    global_derived_values(E);

    if (E->control.verbose)  {
      sprintf(output_file,"%s.info.%d",E->control.data_file,E->parallel.me);
      E->fp_out = fopen(output_file,"w");
	if (E->fp_out == NULL) {
          fprintf(E->fp,"(Instructions.c #1) Cannot open %s\n",output_file);
          exit(8);
	}
      }

    parallel_processor_setup(E);   /* get # of proc in x,y,z */
    parallel_domain_decomp0(E);  /* get local nel, nno, elx, nox et al */

    allocate_common_vars(E);                 
    (E->problem_allocate_vars)(E);
    (E->solver_allocate_vars)(E);
 
           /* logical domain */
    construct_ien(E);
    parallel_domain_boundary_nodes(E);

           /* physical domain */
    node_locations (E);             

    if(E->control.tracer==1)
      (E->problem_tracer_setup)(E);

    allocate_velocity_vars(E);   
              
    get_initial_elapsed_time(E);  /* Get elapsed time from restart run*/
    set_starting_age(E);  /* set the starting age to elapsed time, if desired */
    set_elapsed_time(E);         /* reset to elapsed time to zero, if desired */

    (E->problem_boundary_conds)(E);

    check_bc_consistency(E);

    construct_masks(E);		/* order is important here */
    construct_id(E);
    construct_lm(E);

    parallel_communication_routs_v(E);
    parallel_communication_routs_s(E);

    construct_sub_element(E);
    construct_shape_functions(E);

/*    construct_c3x3matrix(E);       */  /* this matrix results from spherical geometry*/

    mass_matrix(E);

    if (E->parallel.me==0) fprintf(stderr,"time=%f\n",CPU_time0()-start_time);

    construct_surf_det (E); 

    set_sphere_harmonics (E);

    if(E->control.mat_control)
      read_mat_from_file(E);
    else
      construct_mat_group(E);

    (E->problem_initial_fields)(E);   /* temperature/chemistry/melting etc */
    common_initial_fields(E);  /* velocity/pressure/viscosity (viscosity must be done LAST) */

    shutdown_parser(E);
 
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

  E->P[j]	 = (double *) malloc((npno+1)*sizeof(double));
  E->T[j]        = (double *) malloc((nno+1)*sizeof(double));
  E->NP[j]       = (float *) malloc((nno+1)*sizeof(float));
  E->edot[j]     = (float *) malloc((nno+1)*sizeof(float));

  E->Fas410[j]   = (float *) malloc((nno+1)*sizeof(float));
  E->Fas410_b[j] = (float *) malloc((nsf+1)*sizeof(float));
  E->Fas670[j]   = (float *) malloc((nno+1)*sizeof(float));
  E->Fas670_b[j] = (float *) malloc((nsf+1)*sizeof(float));
  E->Fascmb[j]   = (float *) malloc((nno+1)*sizeof(float));
  E->Fascmb_b[j] = (float *) malloc((nsf+1)*sizeof(float));

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
   
  E->mat[j] = (int *) malloc((nel+2)*sizeof(int));
  E->VIP[j] = (float *) malloc((nel+2)*sizeof(float));

  nxyz = max(nox*noz,nox*noy);
  nxyz = 2*max(nxyz,noz*noy);

  E->sien[j]         = (struct SIEN *) malloc((nxyz+2)*sizeof(struct SIEN));
  E->surf_element[j] = (int *) malloc((nxyz+2)*sizeof(int));
  E->surf_node[j]    = (int *) malloc((nsf+2)*sizeof(int));

  }         /* end for cap j  */

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
      E->SinCos[i][j][d]  = (float *)  malloc((nno+1)*sizeof(float));

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

 for(i=0;i<=E->sphere.llmax;i++)
  E->sphere.hindex[i] = (int *)  malloc((E->sphere.llmax+3)*sizeof(int));


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
    E->ELEMENT[i][j] = (unsigned int *) malloc ((nxyz+2)*sizeof(unsigned int));

    for (k=1;k<=nxyz;k++)
       E->ELEMENT[i][j][k] = 0;


    E->elt_del[i][j]=(struct EG *)  malloc((nel+1)*sizeof(struct EG));

    E->EVI[i][j] = (float *)        malloc((nel+2)*vpoints[E->mesh.nsd]*sizeof(float));
    E->BPI[i][j]    = (double *)    malloc((npno+1)*sizeof(double));

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
  
  for(i=1;i<E->lmesh.nel;i++)   {
      E->mat[j][i]=1;
      E->VIP[j][i]=1.0;
  }

  for(i=1;i<=E->lmesh.npno;i++)
      E->P[j][i] = 0.0;

  set_up_nonmg_aliases(E,j); 

  }         /* end for cap j  */




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
    E->temp1[j] = (double *) malloc((E->lmesh.neq+1)*sizeof(double));
    E->F[j] = (double *) malloc((E->lmesh.neq+1)*sizeof(double));
    E->U[j] = (double *) malloc((E->lmesh.neq+2)*sizeof(double));
    E->u1[j] = (double *) malloc((E->lmesh.neq+2)*sizeof(double));

 
    for(i=1;i<=E->mesh.nsd;i++) {
      E->sphere.cap[j].V[i] = (float *) malloc((E->lmesh.nnov+1)*sizeof(float));
      E->sphere.cap[j].VB[i] = (float *)malloc((E->lmesh.nnov+1)*sizeof(float));
      }

    for(i=0;i<=E->lmesh.neq;i++)
      E->U[j][i] = E->temp[j][i] = E->temp1[j][i] = 0.0;

    if(E->control.tracer==1)  {
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

      E->BI[l][j] = (double *) malloc((E->lmesh.NEQ[l]+2)*sizeof(double)); 
      k = (E->lmesh.NOX[l]*E->lmesh.NOZ[l]+E->lmesh.NOX[l]*E->lmesh.NOY[l]+
	  E->lmesh.NOY[l]*E->lmesh.NOZ[l])*6;
      E->zero_resid[l][j] = (int *) malloc((k+2)*sizeof(int)); 
      E->parallel.Skip_id[l][j] = (int *) malloc((k+2)*sizeof(int)); 

      for(i=0;i<E->lmesh.NEQ[l]+2;i++) {
         E->BI[l][j][i]=0.0;
         }

      }   /* end for j & l */
 
  return;  
 }
  

/*  =========================================================  */


void interuption()
    
{  if (Emergency_stop++) exit(0);
   fprintf(stderr,"Cleaning up before exit\n");
   return;
   }


void global_default_values(E)
     struct All_variables *E;
{
    FILE *fp;

  /* FIRST: values which are not changed routinely by the user */
 		
  E->control.v_steps_low = 10;
  E->control.v_steps_upper = 1;
  E->control.max_res_red_each_p_mg = 1.0e-3;
  E->control.accuracy = 1.0e-6;
  E->control.vaccuracy = 1.0e-8;
  E->control.true_vcycle=0;
  E->control.depth_dominated=0;
  E->control.eqn_zigzag=0;
  E->control.verbose=0; /* debugging/profiles */

  /* SECOND: values for which an obvious default setting is useful */

  E->control.ORTHO = 1; /* for orthogonal meshes by default */
  E->control.ORTHOZ = 1; /* for orthogonal meshes by default */

 
    E->control.KERNEL = 0;
    E->control.stokes=0;
    E->control.restart=0;
    E->control.CONVECTION = 0;
    E->control.SLAB = 0;
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

    /* Default: all optional modules set to `off' */
    E->control.MELTING_MODULE = 0;
    E->control.CHEMISTRY_MODULE = 0;

    E->control.GRID_TYPE=1;
    E->mesh.hwidth[1]=E->mesh.hwidth[2]=E->mesh.hwidth[3]=1.0; /* divide by this one ! */
    E->mesh.magnitude[1]=E->mesh.magnitude[2]=E->mesh.magnitude[3]=0.0;
    E->mesh.offset[1]=E->mesh.offset[2]=E->mesh.offset[3]=0.0;

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

  E->viscosity.guess = 0;
  sprintf(E->viscosity.old_file,"initialize");
 
  E->control.precondition = 0;	/* for larger visc contrasts turn this back on  */
  E->control.vprecondition = 1;	
 
  E->mesh.toptbc = 1; /* fixed t */
  E->mesh.bottbc = 1;
  E->mesh.topvbc = 0; /* stress */
  E->mesh.botvbc = 0;
  E->mesh.sidevbc=0;
  E->mesh.periodic_x=0; /* reflection is default*/
  E->mesh.periodic_y=0;
  E->control.VBXtopval=0.0;
  E->control.VBYtopval=0.0;
  E->control.VBXbotval=0.0;
  E->control.VBYbotval=0.0;

  E->crust.total_node = 0;
  E->control.crust = 0;

  E->data.layer_km = 2890.0; /* Earth, whole mantle defaults */
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
  E->data.surf_temp = 0.0;
  E->data.youngs_mod = 1.0e11;
  E->data.Te = 0.0;
  E->data.T_sol0 = 1373.0;	/* Dave's values 1991 (for the earth) */
  E->data.Tsurf = 273.0;
  E->data.dTsol_dz = 3.4e-3 ;
  E->data.dTsol_dF = 440.0;
  E->data.dT_dz = 0.48e-3;
  E->data.delta_S = 250.0;
  E->data.ref_temperature = 2 * 1350.0; /* fixed temperature ... delta T */
    
  /* THIRD: you forgot and then went home, let's see if we can help out */
 
    sprintf(E->control.data_file,"citcom.tmp.%d",getpid());
  
    E->control.NASSEMBLE = 0;
  
    E->mesh.layer[1] =  E->mesh.layer[2] =  E->mesh.layer[3] = 1.0;
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


void common_initial_fields(E)
    struct All_variables *E;
{
    void initial_pressure();
    void initial_velocity();
    void read_viscosity_option();

    report(E,"Initialize pressure field");
    initial_pressure(E);
    report(E,"Initialize velocity field");
    initial_velocity(E);
    report(E,"Initialize viscosity field");
    get_viscosity_option(E);

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
