/*   Functions which solve the heat transport equations using Petrov-Galerkin
     streamline-upwind methods. The process is basically as described in Alex
     Brooks PhD thesis (Caltech) which refers back to Hughes, Liu and Brooks.  */

#include <sys/types.h>
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"

extern int Emergency_stop;

struct el { double gpt[9]; };

static double *DTdot[NCS], *T1[NCS], *Tdot1[NCS], T_interior1;
static double time0,time1,T_interior1;

/* ============================================
   Generic adv-diffusion for temperature field.
   ============================================ */


void advection_diffusion_parameters(E)
     struct All_variables *E;

{

    void std_timestep();
    /* Set intial values, defaults & read parameters*/
    int m=E->parallel.me;

    E->advection.temp_iterations = 2; /* petrov-galerkin iterations: minimum value. */
    E->advection.total_timesteps = 1; 
    E->advection.sub_iterations = 1;
    E->advection.last_sub_iterations = 1;
    E->advection.gamma = 0.5;
    E->advection.dt_reduced = 1.0;         

    E->monitor.T_maxvaried = 1.05;
 
    input_boolean("ADV",&(E->advection.ADVECTION),"on",m);
   
    input_int("minstep",&(E->advection.min_timesteps),"1",m);
    input_int("maxstep",&(E->advection.max_timesteps),"1000",m);
    input_int("maxtotstep",&(E->advection.max_total_timesteps),"1000000",m);
    input_float("finetunedt",&(E->advection.fine_tune_dt),"0.9",m);
    input_float("fixed_timestep",&(E->advection.fixed_timestep),"0.0",m);
    input_int("adv_sub_iterations",&(E->advection.temp_iterations),"2,2,nomax",m);
    input_float("maxadvtime",&(E->advection.max_dimensionless_time),"10.0",m);
   
/*     input_float("sub_tolerance",&(E->advection.vel_substep_aggression),"0.005",m);   */
/*     input_int("maxsub",&(E->advection.max_substeps),"25",m); */
  
/*     input_float("liddefvel",&(E->advection.lid_defining_velocity),"0.01",m); */
/*     input_float("sublayerfrac",&(E->advection.sub_layer_sample_level),"0.5",m);             */
	      

  return;
}

void advection_diffusion_allocate_memory(E)
     struct All_variables *E;

{ int i,m;

  for(m=1;m<=E->sphere.caps_per_proc;m++)  {
    E->Tdot[m]= (double *)malloc((E->lmesh.nno+1)*sizeof(double));
    
    for(i=1;i<=E->lmesh.nno;i++)
      E->Tdot[m][i]=0.0;
    }

return;
}

void PG_timestep_init(struct All_variables *E)
{ 

  void std_timestep();
  double Tmaxd();
  double CPU_time0();
  int m,i;
  static int been_here = 0;

  for(m=1;m<=E->sphere.caps_per_proc;m++)  {
    DTdot[m]= (double *)malloc((E->lmesh.nno+1)*sizeof(double));
    T1[m]= (double *)malloc((E->lmesh.nno+1)*sizeof(double));
    Tdot1[m]= (double *)malloc((E->lmesh.nno+1)*sizeof(double));
  } 

  if (been_here++ ==0)    {
    E->advection.timesteps=0;
  }

  E->advection.timesteps++;
 
  std_timestep(E);

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for (i=1;i<=E->lmesh.nno;i++)   {
      T1[m][i] = E->T[m][i];
      Tdot1[m][i] = E->Tdot[m][i];
    }

  /* get the max temperature for old T */
  T_interior1 = Tmaxd(E,E->T);

  E->advection.dt_reduced = 1.0;         
  E->advection.last_sub_iterations = 1;

  time1= CPU_time0();   

  return;
}

void PG_timestep_solve(struct All_variables *E)
{
  void predictor();
  void corrector();
  void pg_solver();
  void temperatures_conform_bcs();
  double Tmaxd();
  int i,m,psc_pass,iredo;

  do {
    E->advection.timestep *= E->advection.dt_reduced; 
    
    iredo = 0;
    predictor(E,E->T,E->Tdot);
    
    for(psc_pass=0;psc_pass<E->advection.temp_iterations;psc_pass++)   {
      pg_solver(E,E->T,E->Tdot,DTdot,E->convection.heat_sources,E->control.inputdiff,1,E->node);
      corrector(E,E->T,E->Tdot,DTdot);
      temperatures_conform_bcs(E); 
    }	     
    
    /* get the max temperature for new T */
    E->monitor.T_interior = Tmaxd(E,E->T);
    
    if (E->monitor.T_interior/T_interior1 > E->monitor.T_maxvaried) {
      for(m=1;m<=E->sphere.caps_per_proc;m++)
	for (i=1;i<=E->lmesh.nno;i++)   {
	  E->T[m][i] = T1[m][i];
	  E->Tdot[m][i] = Tdot1[m][i];
	}
      iredo = 1;
      E->advection.dt_reduced *= 0.5;         
      E->advection.last_sub_iterations ++;
    }
    
  }  while ( iredo==1 && E->advection.last_sub_iterations <= 5);
  time0= CPU_time0()-time1;
  
  /*     if(E->control.verbose) */
  /*       fprintf(E->fp_out,"time=%f\n",time0); */
  
  E->advection.total_timesteps++;
  E->monitor.elapsed_time += E->advection.timestep; 
  
  return;
}

void PG_timemarching_control(struct All_variables *E)
{ 

  if(((E->advection.total_timesteps < E->advection.max_total_timesteps) &&
      (E->advection.timesteps < E->advection.max_timesteps) && 
      (E->monitor.elapsed_time < E->advection.max_dimensionless_time) ) ||
     (E->advection.total_timesteps < E->advection.min_timesteps) )
    E->control.keep_going = 1;
  else
    E->control.keep_going = 0;
  
  if (E->advection.last_sub_iterations==5)
    E->control.keep_going = 0;

  return;
}

void PG_timestep_fini(struct All_variables *E)
{ 
  int m;
 
  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    free((void *) DTdot[m] );  
    free((void *) T1[m] );   
    free((void *) Tdot1[m] );
  }

  return;
}

void PG_timestep(struct All_variables *E)
{    
    void timestep();
    void predictor();
    void corrector();
    void pg_solver();
    void std_timestep();
    void temperatures_conform_bcs();
    void vcopy();
    double Tmaxd(),global_tdot_d(),aaa,T_interior1;

    int i,j,psc_pass,count,steps,iredo,m;
    int keep_going;

    double *DTdot[NCS], *T1[NCS], *Tdot1[NCS];
    double CPU_time0(),time1,time0,time2;

    static int loops_since_new_eta = 0;
    static int been_here = 0;
   
  for(m=1;m<=E->sphere.caps_per_proc;m++)  {
    DTdot[m]= (double *)malloc((E->lmesh.nno+1)*sizeof(double));
    T1[m]= (double *)malloc((E->lmesh.nno+1)*sizeof(double));
    Tdot1[m]= (double *)malloc((E->lmesh.nno+1)*sizeof(double));
    }

  if (been_here++ ==0)    {
    E->advection.timesteps=0;
  }

  E->advection.timesteps++;
 
  std_timestep(E);

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for (i=1;i<=E->lmesh.nno;i++)   {
       T1[m][i] = E->T[m][i];
       Tdot1[m][i] = E->Tdot[m][i];
       }

  /* get the max temperature for old T */
  T_interior1 = Tmaxd(E,E->T);

  E->advection.dt_reduced = 1.0;         
  E->advection.last_sub_iterations = 1;
    
  time1= CPU_time0();
  do {

    E->advection.timestep *= E->advection.dt_reduced; 

    iredo = 0;
    if (E->advection.ADVECTION) {

      predictor(E,E->T,E->Tdot);
	 
      for(psc_pass=0;psc_pass<E->advection.temp_iterations;psc_pass++)   {
	pg_solver(E,E->T,E->Tdot,DTdot,E->convection.heat_sources,E->control.inputdiff,1,E->node);
	corrector(E,E->T,E->Tdot,DTdot);
	temperatures_conform_bcs(E); 
      }	     
    }

    /* get the max temperature for new T */
    E->monitor.T_interior = Tmaxd(E,E->T);

    if (E->monitor.T_interior/T_interior1 > E->monitor.T_maxvaried) {
      for(m=1;m<=E->sphere.caps_per_proc;m++)
	for (i=1;i<=E->lmesh.nno;i++)   {
	  E->T[m][i] = T1[m][i];
	  E->Tdot[m][i] = Tdot1[m][i];
	}
      iredo = 1;
      E->advection.dt_reduced *= 0.5;         
      E->advection.last_sub_iterations ++;
    }
    
  }  while ( iredo==1 && E->advection.last_sub_iterations <= 5);

  time0= CPU_time0()-time1;

/*     if(E->control.verbose) */
/*       fprintf(E->fp_out,"time=%f\n",time0); */

  E->advection.total_timesteps++;
  E->monitor.elapsed_time += E->advection.timestep; 


  if(((E->advection.total_timesteps < E->advection.max_total_timesteps) &&
      (E->advection.timesteps < E->advection.max_timesteps) && 
      (E->monitor.elapsed_time < E->advection.max_dimensionless_time) ) ||
     (E->advection.total_timesteps < E->advection.min_timesteps) )
    E->control.keep_going = 1;
  else
    E->control.keep_going = 0;

  if (E->advection.last_sub_iterations==5)
    E->control.keep_going = 0;

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    free((void *) DTdot[m] );  
    free((void *) T1[m] );   
    free((void *) Tdot1[m] );
  }
    
  return;  
}


/* ==============================
   predictor and corrector steps.
   ============================== */

void predictor(E,field,fielddot)
     struct All_variables *E;
     double **field,**fielddot;

{ 
  int node,m;
  double multiplier;

  multiplier = (1.0-E->advection.gamma) * E->advection.timestep;

  for (m=1;m<=E->sphere.caps_per_proc;m++) 
    for(node=1;node<=E->lmesh.nno;node++)  {
      field[m][node] += multiplier * fielddot[m][node] ;
      fielddot[m][node] = 0.0;
    }

  return; 
}

void corrector(E,field,fielddot,Dfielddot)
     struct All_variables *E;
     double **field,**fielddot,**Dfielddot;
    
{
  int node,m;
  double multiplier;

  multiplier = E->advection.gamma * E->advection.timestep;
   
  for (m=1;m<=E->sphere.caps_per_proc;m++) 
    for(node=1;node<=E->lmesh.nno;node++) {
      field[m][node] += multiplier * Dfielddot[m][node];
      fielddot[m][node] +=  Dfielddot[m][node]; 
    }
   
  return;  
 }

/* ===================================================
   The solution step -- determine residual vector from
   advective-diffusive terms and solve for delta Tdot
   Two versions are available -- one for Cray-style 
   vector optimizations etc and one optimized for 
   workstations.
   =================================================== */


void pg_solver(E,T,Tdot,DTdot,Q0,diff,bc,FLAGS)
     struct All_variables *E;
     double **T,**Tdot,**DTdot;
     struct SOURCES Q0;
     double diff;
     int bc;
     unsigned int **FLAGS;
{
    void get_global_shape_fn();
    void pg_shape_fn();
    void element_residual();
    void exchange_node_f();
    void exchange_node_d();
    void velo_from_element();

    int el,e,a,i,a1,m;
    double Eres[9],rtf[4][9];  /* correction to the (scalar) Tdot field */
    float VV[4][9];

    struct Shape_function PG;
    struct Shape_function GN;
    struct Shape_function_dA dOmega;
    struct Shape_function_dx GNx;
 
    const int dims=E->mesh.nsd;
    const int dofs=E->mesh.dof;
    const int ends=enodes[dims];
    const int sphere_key = 0;
  
    for (m=1;m<=E->sphere.caps_per_proc;m++) 
      for(i=1;i<=E->lmesh.nno;i++)
 	 DTdot[m][i] = 0.0;

    for (m=1;m<=E->sphere.caps_per_proc;m++) 
       for(el=1;el<=E->lmesh.nel;el++)    {

          velo_from_element(E,VV,m,el,sphere_key);

	  get_global_shape_fn(E,el,&GN,&GNx,&dOmega,0,sphere_key,rtf,E->mesh.levmax,m);

          pg_shape_fn(E,el,&PG,&GNx,VV,diff,m);
          element_residual(E,el,PG,GNx,dOmega,VV,T,Tdot,Q0,Eres,diff,E->sphere.cap[m].TB,FLAGS,m);

        for(a=1;a<=ends;a++) {
	    a1 = E->ien[m][el].node[a];
	    DTdot[m][a1] += Eres[a]; 
           }

        } /* next element */

    exchange_node_d(E,DTdot,E->mesh.levmax);

    for (m=1;m<=E->sphere.caps_per_proc;m++) 
      for(i=1;i<=E->lmesh.nno;i++) {
        if(!(E->node[m][i] & (TBX | TBY | TBZ)))
	  DTdot[m][i] *= E->Mass[m][i];         /* lumped mass matrix */
	else
	  DTdot[m][i] = 0.0;         /* lumped mass matrix */
      }

    return;    
}



/* ===================================================
   Petrov-Galerkin shape functions for a given element
   =================================================== */

void pg_shape_fn(E,el,PG,GNx,VV,diffusion,m)
     struct All_variables *E;
     int el,m;
     struct Shape_function *PG;
     struct Shape_function_dx *GNx;
     float VV[4][9];
     double diffusion;

{ 
    int i,j,node;
    int *ienm;

    double uc1,uc2,uc3;
    double u1,u2,u3;
    double aa,bb,cc,uxse,ueta,ufai,xse,eta,fai,dx1,dx2,dx3,adiff,rr1;

    double prod1,unorm,twodiff;

    const int dims=E->mesh.nsd;
    const int dofs=E->mesh.dof;
    const int lev=E->mesh.levmax;
    const int nno=E->lmesh.nno;
    const int ends=enodes[E->mesh.nsd];
    const int vpts=vpoints[E->mesh.nsd];
  
    ienm=E->ien[m][el].node;

    twodiff = 2.0*diffusion;
 
    uc1 =  uc2 = uc3 = 0.0;

    for(i=1;i<=ENODES3D;i++) {
      uc1 +=  E->N.ppt[GNPINDEX(i,1)]*VV[1][i];
      uc2 +=  E->N.ppt[GNPINDEX(i,1)]*VV[2][i];
      uc3 +=  E->N.ppt[GNPINDEX(i,1)]*VV[3][i];
      }
  
    dx1=0.25*(E->x[m][1][ienm[3]]+E->x[m][1][ienm[4]]
             +E->x[m][1][ienm[7]]+E->x[m][1][ienm[8]]
             -E->x[m][1][ienm[1]]-E->x[m][1][ienm[2]]
             -E->x[m][1][ienm[5]]-E->x[m][1][ienm[6]]);
    dx2=0.25*(E->x[m][2][ienm[3]]+E->x[m][2][ienm[4]]
             +E->x[m][2][ienm[7]]+E->x[m][2][ienm[8]]
             -E->x[m][2][ienm[1]]-E->x[m][2][ienm[2]]
             -E->x[m][2][ienm[5]]-E->x[m][2][ienm[6]]);
    dx3=0.25*(E->x[m][3][ienm[3]]+E->x[m][3][ienm[4]]
             +E->x[m][3][ienm[7]]+E->x[m][3][ienm[8]]
             -E->x[m][3][ienm[1]]-E->x[m][3][ienm[2]]
             -E->x[m][3][ienm[5]]-E->x[m][3][ienm[6]]);
    uxse = fabs(uc1*dx1+uc2*dx2+uc3*dx3);

    dx1=0.25*(E->x[m][1][ienm[2]]+E->x[m][1][ienm[3]]
             +E->x[m][1][ienm[6]]+E->x[m][1][ienm[7]]
             -E->x[m][1][ienm[1]]-E->x[m][1][ienm[4]]
             -E->x[m][1][ienm[5]]-E->x[m][1][ienm[8]]);
    dx2=0.25*(E->x[m][2][ienm[2]]+E->x[m][2][ienm[3]]
             +E->x[m][2][ienm[6]]+E->x[m][2][ienm[7]]
             -E->x[m][2][ienm[1]]-E->x[m][2][ienm[4]]
             -E->x[m][2][ienm[5]]-E->x[m][2][ienm[8]]);
    dx3=0.25*(E->x[m][3][ienm[2]]+E->x[m][3][ienm[3]]
             +E->x[m][3][ienm[6]]+E->x[m][3][ienm[7]]
             -E->x[m][3][ienm[1]]-E->x[m][3][ienm[4]]
             -E->x[m][3][ienm[5]]-E->x[m][3][ienm[8]]);
    ueta = fabs(uc1*dx1+uc2*dx2+uc3*dx3);

    dx1=0.25*(E->x[m][1][ienm[5]]+E->x[m][1][ienm[6]]
             +E->x[m][1][ienm[7]]+E->x[m][1][ienm[8]]
             -E->x[m][1][ienm[1]]-E->x[m][1][ienm[2]]
             -E->x[m][1][ienm[3]]-E->x[m][1][ienm[4]]);
    dx2=0.25*(E->x[m][2][ienm[5]]+E->x[m][2][ienm[6]]
             +E->x[m][2][ienm[7]]+E->x[m][2][ienm[8]]
             -E->x[m][2][ienm[1]]-E->x[m][2][ienm[2]]
             -E->x[m][2][ienm[3]]-E->x[m][2][ienm[4]]);
    dx3=0.25*(E->x[m][3][ienm[5]]+E->x[m][3][ienm[6]]
             +E->x[m][3][ienm[7]]+E->x[m][3][ienm[8]]
             -E->x[m][3][ienm[1]]-E->x[m][3][ienm[2]]
             -E->x[m][3][ienm[3]]-E->x[m][3][ienm[4]]);
    ufai = fabs(uc1*dx1+uc2*dx2+uc3*dx3);

/*    xse = (uxse>twodiff)? (1.0-twodiff/uxse):0.0;
    eta = (ueta>twodiff)? (1.0-twodiff/ueta):0.0;
    fai = (ufai>twodiff)? (1.0-twodiff/ufai):0.0;
*/

    aa = 2.0*uxse/twodiff;
    bb = 2.0*ueta/twodiff;
    cc = 2.0*ufai/twodiff;
    xse = (1.0+exp(-aa))/(1.0-exp(-aa))-twodiff/uxse;
    eta = (1.0+exp(-bb))/(1.0-exp(-bb))-twodiff/ueta;
    fai = (1.0+exp(-cc))/(1.0-exp(-cc))-twodiff/ufai;

    unorm = uc1*uc1 + uc2*uc2 + uc3*uc3;

    adiff = (unorm>0.000001)?( (uxse*xse+ueta*eta+ufai*fai)/(2.0*unorm) ):0.0;

    for(i=1;i<=VPOINTS3D;i++) {
       u1 = u2 = u3 = 0.0;
       for(j=1;j<=ENODES3D;j++)  /* this line heavily used */ {
		u1 += VV[1][j] * E->N.vpt[GNVINDEX(j,i)]; 
		u2 += VV[2][j] * E->N.vpt[GNVINDEX(j,i)];
	   	u3 += VV[3][j] * E->N.vpt[GNVINDEX(j,i)];
	    }
	    
       for(j=1;j<=ENODES3D;j++) {
            prod1 = (u1 * GNx->vpt[GNVXINDEX(0,j,i)] +
                     u2 * GNx->vpt[GNVXINDEX(1,j,i)] +
                    u3 * GNx->vpt[GNVXINDEX(2,j,i)] ) ;
		
	    PG->vpt[GNVINDEX(j,i)] = E->N.vpt[GNVINDEX(j,i)] + adiff * prod1;
	    }
       }
    
   return;
 }



/* ==========================================
   Residual force vector from heat-transport.
   Used to correct the Tdot term.
   =========================================  */

void element_residual(E,el,PG,GNx,dOmega,VV,field,fielddot,Q0,Eres,diff,BC,FLAGS,m)
     struct All_variables *E;
     int el,m;
     struct Shape_function PG;
     struct Shape_function_dA dOmega;
     struct Shape_function_dx GNx;
     float VV[4][9];
     double **field,**fielddot;
     struct SOURCES Q0;
     double Eres[9];
     double diff;
     float **BC;
     unsigned int **FLAGS;

{
    int i,j,a,k,node,nodes[5],d,aid,back_front,onedfns;
    double Q;
    double dT[9];
    double tx1[9],tx2[9],tx3[9];
    double v1[9],v2[9],v3[9];
    double adv_dT,t2[4];
    double T,DT;
 
    register double prod,sfn;
    struct Shape_function1 GM;
    struct Shape_function1_dA dGamma;
    double temp;

    void get_global_1d_shape_fn();
 
    const int dims=E->mesh.nsd;
    const int dofs=E->mesh.dof;
    const int nno=E->lmesh.nno;
    const int lev=E->mesh.levmax;
    const int ends=enodes[dims];
    const int vpts=vpoints[dims];
    const int diffusion = (diff != 0.0); 
 
    for(i=1;i<=vpts;i++)	{ 
      dT[i]=0.0;
      v1[i] = tx1[i]=  0.0;
      v2[i] = tx2[i]=  0.0;
      v3[i] = tx3[i]=  0.0;
      }

    for(j=1;j<=ends;j++)       {
      node = E->ien[m][el].node[j];
      T = field[m][node];  
      if(E->node[m][node] & (TBX | TBY | TBZ))
	    DT=0.0;
      else
	    DT = fielddot[m][node];
	
      for(i=1;i<=vpts;i++)  {
		  dT[i] += DT * E->N.vpt[GNVINDEX(j,i)];
		  tx1[i] += GNx.vpt[GNVXINDEX(0,j,i)] * T; 
		  tx2[i] += GNx.vpt[GNVXINDEX(1,j,i)] * T; 
	 	  tx3[i] += GNx.vpt[GNVXINDEX(2,j,i)] * T; 
		  sfn = E->N.vpt[GNVINDEX(j,i)];
		  v1[i] += VV[1][j] * sfn;
		  v2[i] += VV[2][j] * sfn;
		  v3[i] += VV[3][j] * sfn;     
	      }
      }

/*    Q=0.0;
    for(i=0;i<Q0.number;i++)
	  Q += Q0.Q[i] * exp(-Q0.lambda[i] * (E->monitor.elapsed_time+Q0.t_offset));
*/

    Q = E->control.Q0;

    /* construct residual from this information */


    if(diffusion){
      for(j=1;j<=ends;j++) {
	Eres[j]=0.0;
	for(i=1;i<=vpts;i++) 
	  Eres[j] -=
	    PG.vpt[GNVINDEX(j,i)] * dOmega.vpt[i] 
	    * (dT[i] - Q + v1[i]*tx1[i] + v2[i]*tx2[i] + v3[i]*tx3[i])
 	    + diff*dOmega.vpt[i] * (GNx.vpt[GNVXINDEX(0,j,i)]*tx1[i] +
				    GNx.vpt[GNVXINDEX(1,j,i)]*tx2[i] +
				    GNx.vpt[GNVXINDEX(2,j,i)]*tx3[i] ); 
      }
    }

    else { /* no diffusion term */
      for(j=1;j<=ends;j++) {
	Eres[j]=0.0;
	for(i=1;i<=vpts;i++) 
	  Eres[j] -= PG.vpt[GNVINDEX(j,i)] * dOmega.vpt[i] * (dT[i] - Q + v1[i] * tx1[i] + v2[i] * tx2[i] + v3[i] * tx3[i]);
      }
    }	

    /* See brooks etc: the diffusive term is excused upwinding for 
       rectangular elements  */
  
    /* include BC's for fluxes at (nominally horizontal) edges (X-Y plane) */

    if(FLAGS!=NULL) {
      onedfns=0;
      for(a=1;a<=ends;a++)
	if (FLAGS[m][E->ien[m][el].node[a]] & FBZ) {
	  if (!onedfns++) get_global_1d_shape_fn(E,el,&GM,&dGamma,1,m);
 
	  nodes[1] = loc[loc[a].node_nebrs[0][0]].node_nebrs[2][0];
	  nodes[2] = loc[loc[a].node_nebrs[0][1]].node_nebrs[2][0];
	  nodes[4] = loc[loc[a].node_nebrs[0][0]].node_nebrs[2][1];
	  nodes[3] = loc[loc[a].node_nebrs[0][1]].node_nebrs[2][1];
	  
	  for(aid=0,j=1;j<=onedvpoints[E->mesh.nsd];j++)
	    if (a==nodes[j])
	      aid = j;
	  if(aid==0)  
	    printf("%d: mixed up in pg-flux int: looking for %d\n",el,a);

	  if (loc[a].plus[1] != 0)
	    back_front = 0;
	  else back_front = 1;
	  
	  for(j=1;j<=onedvpoints[dims];j++)
	    for(k=1;k<=onedvpoints[dims];k++)
	      Eres[a] += dGamma.vpt[GMVGAMMA(back_front,j)] *
		E->M.vpt[GMVINDEX(aid,j)] * g_1d[j].weight[dims-1] *
		BC[2][E->ien[m][el].node[a]] * E->M.vpt[GMVINDEX(k,j)];
	}
    } 
    
    return; 
}




/* =====================================================
   Obtain largest possible timestep (no melt considered)
   =====================================================  */


void std_timestep(E)
     struct All_variables *E;

{ 
    static int been_here = 0;
    static float diff_timestep,root3,root2;
    int i,d,n,nel,el,node,m;

    float global_fmin();
    void velo_from_element();

    float adv_timestep;
    float ts,uc1,uc2,uc3,uc,size,step,VV[4][9];
    
    const int dims=E->mesh.nsd;
    const int dofs=E->mesh.dof;
    const int nno=E->lmesh.nno;
    const int lev=E->mesh.levmax;
    const int ends=enodes[dims];
    const int sphere_key = 1;
    
    nel=E->lmesh.nel;

    if(E->advection.fixed_timestep != 0.0) {
      E->advection.timestep = E->advection.fixed_timestep;
      return;
    }

    if (been_here == 0)  {
      diff_timestep = 1.0e8; 
      for(m=1;m<=E->sphere.caps_per_proc;m++)
	for(el=1;el<=nel;el++)  { 
	  for(d=1;d<=dims;d++)    {
	    ts = E->eco[m][el].size[d] * E->eco[m][el].size[d];
	    diff_timestep = min(diff_timestep,ts);
	  }
	}
      diff_timestep = global_fmin(E,diff_timestep);

      diff_timestep = 0.5*diff_timestep;

/*      diff_timestep = ((3==dims)? 0.125:0.25) * diff_timestep;  */
    }
    
    adv_timestep = 1.0e8;
    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(el=1;el<=nel;el++) {

	velo_from_element(E,VV,m,el,sphere_key);

	uc=uc1=uc2=uc3=0.0;	  
	for(i=1;i<=ENODES3D;i++) {
	  uc1 += E->N.ppt[GNPINDEX(i,1)]*VV[1][i];
	  uc2 += E->N.ppt[GNPINDEX(i,1)]*VV[2][i];
	  uc3 += E->N.ppt[GNPINDEX(i,1)]*VV[3][i];
        }
	uc = fabs(uc1)/E->eco[m][el].size[1] + fabs(uc2)/E->eco[m][el].size[2] + fabs(uc3)/E->eco[m][el].size[3];

	step = (0.5/uc);
	adv_timestep = min(adv_timestep,step);
      }

    adv_timestep = E->advection.dt_reduced * adv_timestep;         

    adv_timestep = 1.0e-32+min(E->advection.fine_tune_dt*adv_timestep,diff_timestep);
    
    E->advection.timestep = global_fmin(E,adv_timestep);

/*     if (E->parallel.me==0) */
/*       fprintf(stderr, "adv_timestep=%g diff_timestep=%g\n",adv_timestep,diff_timestep); */

    return; 
  }
