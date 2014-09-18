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
/*   Functions which solve the heat transport equations using Petrov-Galerkin
     streamline-upwind methods. The process is basically as described in Alex
     Brooks PhD thesis (Caltech) which refers back to Hughes, Liu and Brooks.  */

#include <sys/types.h>

#include "element_definitions.h"
#include "global_defs.h"
#include <math.h>
#include "advection_diffusion.h"
#include "parsing.h"

static void set_diffusion_timestep(struct All_variables *E);
static void predictor(struct All_variables *E, double **field,
                      double **fielddot);
static void corrector(struct All_variables *E, double **field,
                      double **fielddot, double **Dfielddot);
static void pg_solver(struct All_variables *E,
                      double **T, double **Tdot, double **DTdot,
                      struct SOURCES *Q0,
                      double diff, int bc, unsigned int **FLAGS);
static void pg_shape_fn(struct All_variables *E, int el,
                        struct Shape_function *PG,
                        struct Shape_function_dx *GNx,
                        float VV[4][9], double rtf[4][9],
                        double diffusion);
static void element_residual(struct All_variables *E, int el,
                             struct Shape_function *PG,
                             struct Shape_function_dx *GNx,
                             struct Shape_function_dA *dOmega,
                             float VV[4][9],
                             double **field, double **fielddot,
                             struct SOURCES *Q0,
                             double Eres[9], double rtf[4][9],
                             double diff, float **BC,
                             unsigned int **FLAGS);
static void filter(struct All_variables *E);
static void process_heating(struct All_variables *E, int psc_pass);

/* ============================================
   Generic adv-diffusion for temperature field.
   ============================================ */


/***************************************************************/

void advection_diffusion_parameters(struct All_variables *E)
{

    /* Set intial values, defaults & read parameters*/
    int m=E->parallel.me;

    input_boolean("ADV",&(E->advection.ADVECTION),"on",m);
    input_boolean("filter_temp",&(E->advection.filter_temperature),"off",m);
    input_boolean("monitor_max_T",&(E->advection.monitor_max_T),"on",m);

    input_int("minstep",&(E->advection.min_timesteps),"1",m);
    input_int("maxstep",&(E->advection.max_timesteps),"1000",m);
    input_int("maxtotstep",&(E->advection.max_total_timesteps),"1000000",m);
    input_float("finetunedt",&(E->advection.fine_tune_dt),"0.9",m);
    input_float("fixed_timestep",&(E->advection.fixed_timestep),"0.0",m);
    input_float("adv_gamma",&(E->advection.gamma),"0.5",m);
    input_int("adv_sub_iterations",&(E->advection.temp_iterations),"2,1,nomax",m);

    input_float("inputdiffusivity",&(E->control.inputdiff),"1.0",m);


    return;
}


void advection_diffusion_allocate_memory(struct All_variables *E)
{
  int i,m;

    E->Tdot[CPPR]= (double *)malloc((E->lmesh.nno+1)*sizeof(double));

    for(i=1;i<=E->lmesh.nno;i++)
      E->Tdot[CPPR][i]=0.0;
}


void PG_timestep_init(struct All_variables *E)
{
  set_diffusion_timestep(E);
}


void PG_timestep(struct All_variables *E)
{
    void std_timestep();
    void PG_timestep_solve();

    std_timestep(E);

    PG_timestep_solve(E);
}



/* =====================================================
   Obtain largest possible timestep (no melt considered)
   =====================================================  */


void std_timestep(struct All_variables *E)
{
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

    adv_timestep = 1.0e8;
    for(el=1;el<=nel;el++) {

      velo_from_element(E,VV,el,sphere_key);

      uc=uc1=uc2=uc3=0.0;
      for(i=1;i<=ENODES3D;i++) {
        uc1 += E->N.ppt[GNPINDEX(i,1)]*VV[1][i];
        uc2 += E->N.ppt[GNPINDEX(i,1)]*VV[2][i];
        uc3 += E->N.ppt[GNPINDEX(i,1)]*VV[3][i];
      }
      uc = fabs(uc1)/E->eco[CPPR][el].size[1] + fabs(uc2)/E->eco[CPPR][el].size[2] + fabs(uc3)/E->eco[CPPR][el].size[3];

      step = (0.5/uc);
      adv_timestep = min(adv_timestep,step);
    }

    adv_timestep = E->advection.dt_reduced * adv_timestep;

    adv_timestep = 1.0e-32 + min(E->advection.fine_tune_dt*adv_timestep,
				 E->advection.diff_timestep);

    E->advection.timestep = global_fmin(E,adv_timestep);
}


void PG_timestep_solve(struct All_variables *E)
{

  double Tmaxd();
  void temperatures_conform_bcs();
  void lith_age_conform_tbc();
  void assimilate_lith_conform_bcs();
  int i,m,psc_pass,iredo;
  double time0,time1,T_interior1;
  double *DTdot[NCS], *T1[NCS], *Tdot1[NCS];

  E->advection.timesteps++;

  DTdot[CPPR]= (double *)malloc((E->lmesh.nno+1)*sizeof(double));


  if(E->advection.monitor_max_T) {
     T1[CPPR]= (double *)malloc((E->lmesh.nno+1)*sizeof(double));
     Tdot1[CPPR]= (double *)malloc((E->lmesh.nno+1)*sizeof(double));

     for (i=1;i<=E->lmesh.nno;i++)   {
         T1[CPPR][i] = E->T[CPPR][i];
         Tdot1[CPPR][i] = E->Tdot[CPPR][i];
     }

     /* get the max temperature for old T */
     T_interior1 = Tmaxd(E,E->T);
  }

  E->advection.dt_reduced = 1.0;
  E->advection.last_sub_iterations = 1;


  do {
    E->advection.timestep *= E->advection.dt_reduced;

    iredo = 0;
    if (E->advection.ADVECTION) {

      predictor(E,E->T,E->Tdot);

      for(psc_pass=0;psc_pass<E->advection.temp_iterations;psc_pass++)   {
        /* adiabatic, dissipative and latent heating*/
        if(E->control.disptn_number != 0)
          process_heating(E, psc_pass);

        /* XXX: replace inputdiff with refstate.thermal_conductivity */
	pg_solver(E,E->T,E->Tdot,DTdot,&(E->convection.heat_sources),E->control.inputdiff,1,E->node);
	corrector(E,E->T,E->Tdot,DTdot);
	temperatures_conform_bcs(E);
      }

      if(E->advection.monitor_max_T) {
          /* get the max temperature for new T */
          E->monitor.T_interior = Tmaxd(E,E->T);

          /* if the max temperature changes too much, restore the old
           * temperature field, calling the temperature solver using
           * half of the timestep size */
          if (E->monitor.T_interior/T_interior1 > E->monitor.T_maxvaried) {
              if(E->parallel.me==0) {
                  fprintf(stderr, "max T varied from %e to %e\n",
                          T_interior1, E->monitor.T_interior);
                  fprintf(E->fp, "max T varied from %e to %e\n",
                          T_interior1, E->monitor.T_interior);
              }
              for (i=1;i<=E->lmesh.nno;i++)   {
                  E->T[CPPR][i] = T1[CPPR][i];
                  E->Tdot[CPPR][i] = Tdot1[CPPR][i];
              }
              iredo = 1;
              E->advection.dt_reduced *= 0.5;
              E->advection.last_sub_iterations ++;
          }
      }
    }

  }  while ( iredo==1 && E->advection.last_sub_iterations <= 5);


  /* filter temperature to remove over-/under-shoot */
  if(E->advection.filter_temperature)
    filter(E);


  E->advection.total_timesteps++;
  E->monitor.elapsed_time += E->advection.timestep;

  if (E->advection.last_sub_iterations==5)
    E->control.keep_going = 0;

  free((void *) DTdot[CPPR] );

  if(E->advection.monitor_max_T) {
    free((void *) T1[CPPR] );
    free((void *) Tdot1[CPPR] );
  }

  if(E->control.lith_age) {
      if(E->parallel.me==0) fprintf(stderr,"PG_timestep_solve\n");
      lith_age_conform_tbc(E);
      assimilate_lith_conform_bcs(E);
  }
}


/***************************************************************/

static void set_diffusion_timestep(struct All_variables *E)
{
  float diff_timestep, ts;
  int m, el, d;

  float global_fmin();

  diff_timestep = 1.0e8;
    for(el=1;el<=E->lmesh.nel;el++)  {
      for(d=1;d<=E->mesh.nsd;d++)    {
	ts = E->eco[CPPR][el].size[d] * E->eco[CPPR][el].size[d];
	diff_timestep = min(diff_timestep,ts);
      }
    }

  diff_timestep = global_fmin(E,diff_timestep);
  E->advection.diff_timestep = 0.5 * diff_timestep;
}


/* ==============================
   predictor and corrector steps.
   ============================== */

static void predictor(struct All_variables *E, double **field, double **fielddot)
{
  int node,m;
  double multiplier;

  multiplier = (1.0-E->advection.gamma) * E->advection.timestep;

  for(node=1;node<=E->lmesh.nno;node++)  {
    field[CPPR][node] += multiplier * fielddot[CPPR][node] ;
    fielddot[CPPR][node] = 0.0;
  }
}


static void corrector(struct All_variables *E, double **field,
                      double **fielddot, double **Dfielddot)
{
  int node,m;
  double multiplier;

  multiplier = E->advection.gamma * E->advection.timestep;

  for(node=1;node<=E->lmesh.nno;node++) {
    field[CPPR][node] += multiplier * Dfielddot[CPPR][node];
    fielddot[CPPR][node] +=  Dfielddot[CPPR][node];
  }
}


/* ===================================================
   The solution step -- determine residual vector from
   advective-diffusive terms and solve for delta Tdot
   Two versions are available -- one for Cray-style
   vector optimizations etc and one optimized for
   workstations.
   =================================================== */


static void pg_solver(struct All_variables *E,
                      double **T, double **Tdot, double **DTdot,
                      struct SOURCES *Q0,
                      double diff, int bc, unsigned int **FLAGS)
{
    void get_rtf_at_vpts();
    void velo_from_element();

    int el,e,a,i,a1,m;
    double Eres[9],rtf[4][9];  /* correction to the (scalar) Tdot field */
    float VV[4][9];

    struct Shape_function PG;

    const int dims=E->mesh.nsd;
    const int dofs=E->mesh.dof;
    const int ends=enodes[dims];
    const int sphere_key = 1;
    const int lev=E->mesh.levmax;

    for(i=1;i<=E->lmesh.nno;i++)
     DTdot[CPPR][i] = 0.0;

   for(el=1;el<=E->lmesh.nel;el++)    {

      velo_from_element(E,VV,el,sphere_key);

      get_rtf_at_vpts(E, lev, el, rtf);

      /* XXX: replace diff with refstate.thermal_conductivity */
      pg_shape_fn(E, el, &PG, &(E->gNX[CPPR][el]), VV, rtf, diff);
      element_residual(E, el, &PG, &(E->gNX[CPPR][el]), &(E->gDA[CPPR][el]),
                       VV, T, Tdot,
                       Q0, Eres, rtf, diff, E->sphere.cap[CPPR].TB,
                       FLAGS);

      for(a=1;a<=ends;a++) {
        a1 = E->ien[CPPR][el].node[a];
        DTdot[CPPR][a1] += Eres[a];
      }

    } /* next element */

    (E->exchange_node_d)(E,DTdot,lev);

    for(i=1;i<=E->lmesh.nno;i++) {
      if(!(E->node[CPPR][i] & (TBX | TBY | TBZ))){
        DTdot[CPPR][i] *= E->TMass[CPPR][i];         /* lumped mass matrix */
      }	else {
        DTdot[CPPR][i] = 0.0;         /* lumped mass matrix */
      }
    }
}



/* ===================================================
   Petrov-Galerkin shape functions for a given element
   =================================================== */

static void pg_shape_fn(struct All_variables *E, int el,
                        struct Shape_function *PG,
                        struct Shape_function_dx *GNx,
                        float VV[4][9], double rtf[4][9],
                        double diffusion)
{
    int i,j;
    int *ienm;

    double uc1,uc2,uc3;
    double u1,u2,u3,sint[9];
    double uxse,ueta,ufai,xse,eta,fai,adiff;

    double prod1,unorm,twodiff;

    ienm=E->ien[CPPR][el].node;

    twodiff = 2.0*diffusion;

    uc1 =  uc2 = uc3 = 0.0;

    for(i=1;i<=ENODES3D;i++) {
      uc1 +=  E->N.ppt[GNPINDEX(i,1)]*VV[1][i];
      uc2 +=  E->N.ppt[GNPINDEX(i,1)]*VV[2][i];
      uc3 +=  E->N.ppt[GNPINDEX(i,1)]*VV[3][i];
      }

    uxse = fabs(uc1*E->eco[CPPR][el].size[1]);
    ueta = fabs(uc2*E->eco[CPPR][el].size[2]);
    ufai = fabs(uc3*E->eco[CPPR][el].size[3]);

    xse = (uxse>twodiff)? (1.0-twodiff/uxse):0.0;
    eta = (ueta>twodiff)? (1.0-twodiff/ueta):0.0;
    fai = (ufai>twodiff)? (1.0-twodiff/ufai):0.0;


    unorm = uc1*uc1 + uc2*uc2 + uc3*uc3;

    adiff = (unorm>0.000001)?( (uxse*xse+ueta*eta+ufai*fai)/(2.0*unorm) ):0.0;

    for(i=1;i<=VPOINTS3D;i++)
       sint[i] = rtf[3][i]/sin(rtf[1][i]);

    for(i=1;i<=VPOINTS3D;i++) {
       u1 = u2 = u3 = 0.0;
       for(j=1;j<=ENODES3D;j++)  /* this line heavily used */ {
		u1 += VV[1][j] * E->N.vpt[GNVINDEX(j,i)];
		u2 += VV[2][j] * E->N.vpt[GNVINDEX(j,i)];
	   	u3 += VV[3][j] * E->N.vpt[GNVINDEX(j,i)];
	    }

       for(j=1;j<=ENODES3D;j++) {
            prod1 = (u1 * GNx->vpt[GNVXINDEX(0,j,i)]*rtf[3][i] +
                     u2 * GNx->vpt[GNVXINDEX(1,j,i)]*sint[i] +
                     u3 * GNx->vpt[GNVXINDEX(2,j,i)] ) ;

	    PG->vpt[GNVINDEX(j,i)] = E->N.vpt[GNVINDEX(j,i)] + adiff * prod1;
	    }
       }
}



/* ==========================================
   Residual force vector from heat-transport.
   Used to correct the Tdot term.
   =========================================  */

static void element_residual(struct All_variables *E, int el,
                             struct Shape_function *PG,
                             struct Shape_function_dx *GNx,
                             struct Shape_function_dA *dOmega,
                             float VV[4][9],
                             double **field, double **fielddot,
                             struct SOURCES *Q0,
                             double Eres[9], double rtf[4][9],
                             double diff, float **BC,
                             unsigned int **FLAGS)
{
    int i,j,a,k,node,nodes[5],d,aid,back_front,onedfns;
    double Q;
    double dT[9];
    double tx1[9],tx2[9],tx3[9],sint[9];
    double v1[9],v2[9],v3[9];
    double adv_dT,t2[4];
    double T,DT;

    double prod,sfn;
    struct Shape_function1 GM;
    struct Shape_function1_dA dGamma;
    double temp,rho,cp,heating;
    int nz;

    void get_global_1d_shape_fn();

    const int dims=E->mesh.nsd;
    const int dofs=E->mesh.dof;
    const int nno=E->lmesh.nno;
    const int lev=E->mesh.levmax;
    const int ends=enodes[dims];
    const int vpts=vpoints[dims];
    const int onedvpts = onedvpoints[dims];
    const int diffusion = (diff != 0.0);

    for(i=1;i<=vpts;i++)	{
      dT[i]=0.0;
      v1[i] = tx1[i]=  0.0;
      v2[i] = tx2[i]=  0.0;
      v3[i] = tx3[i]=  0.0;
      }

    for(i=1;i<=vpts;i++)
        sint[i] = rtf[3][i]/sin(rtf[1][i]);

    for(j=1;j<=ends;j++)       {
      node = E->ien[CPPR][el].node[j];
      T = field[CPPR][node];
      if(E->node[CPPR][node] & (TBX | TBY | TBZ))
	    DT=0.0;
      else
	    DT = fielddot[CPPR][node];

      for(i=1;i<=vpts;i++)  {
          dT[i] += DT * E->N.vpt[GNVINDEX(j,i)];
          tx1[i] += GNx->vpt[GNVXINDEX(0,j,i)] * T * rtf[3][i];
          tx2[i] += GNx->vpt[GNVXINDEX(1,j,i)] * T * sint[i];
          tx3[i] += GNx->vpt[GNVXINDEX(2,j,i)] * T;
          sfn = E->N.vpt[GNVINDEX(j,i)];
          v1[i] += VV[1][j] * sfn;
          v2[i] += VV[2][j] * sfn;
          v3[i] += VV[3][j] * sfn;
      }
    }

/*    Q=0.0;
    for(i=0;i<Q0.number;i++)
	  Q += Q0->Q[i] * exp(-Q0->lambda[i] * (E->monitor.elapsed_time+Q0->t_offset));
*/

    /* heat production */
    Q = E->control.Q0;

    /* should we add a compositional contribution? */
    if(E->control.tracer_enriched){
      /* XXX: change Q and Q0 to be a vector of ncomp elements */

      /* Q = Q0 for C = 0, Q = Q0ER for C = 1, and linearly in
	 between  */
      Q *= (1.0 - E->composition.comp_el[CPPR][0][el]);
      Q += E->composition.comp_el[CPPR][0][el] * E->control.Q0ER;
    }

    nz = ((el-1) % E->lmesh.elz) + 1;
    rho = 0.5 * (E->refstate.rho[nz] + E->refstate.rho[nz+1]);
    cp = 0.5 * (E->refstate.heat_capacity[nz] + E->refstate.heat_capacity[nz+1]);

    if(E->control.disptn_number == 0)
        heating = rho * Q;
    else
        /* E->heating_latent is actually the inverse of latent heating */
        heating = (rho * Q - E->heating_adi[CPPR][el] + E->heating_visc[CPPR][el])
            * E->heating_latent[CPPR][el];

    /* construct residual from this information */


    if(diffusion){
      for(j=1;j<=ends;j++) {
	Eres[j]=0.0;
	for(i=1;i<=vpts;i++)
	  Eres[j] -=
	    PG->vpt[GNVINDEX(j,i)] * dOmega->vpt[i]
              * ((dT[i] + v1[i]*tx1[i] + v2[i]*tx2[i] + v3[i]*tx3[i])*rho*cp
                 - heating )
              + diff * dOmega->vpt[i] * E->heating_latent[CPPR][el]
              * (GNx->vpt[GNVXINDEX(0,j,i)]*tx1[i]*rtf[3][i] +
                 GNx->vpt[GNVXINDEX(1,j,i)]*tx2[i]*sint[i] +
                 GNx->vpt[GNVXINDEX(2,j,i)]*tx3[i] );
      }
    }

    else { /* no diffusion term */
      for(j=1;j<=ends;j++) {
	Eres[j]=0.0;
	for(i=1;i<=vpts;i++)
	  Eres[j] -= PG->vpt[GNVINDEX(j,i)] * dOmega->vpt[i]
              * (dT[i] - heating + v1[i]*tx1[i] + v2[i]*tx2[i] + v3[i]*tx3[i]);
      }
    }

    /* See brooks etc: the diffusive term is excused upwinding for
       rectangular elements  */

    /* include BC's for fluxes at (nominally horizontal) edges (X-Y plane) */

    if(FLAGS!=NULL) {
      aid = -1;
      if (FLAGS[CPPR][E->ien[CPPR][el].node[1]] & FBZ) {   // only check for the 1st node
          aid = 0;
	  get_global_1d_shape_fn(E,el,&GM,&dGamma,aid);
          }
      else if (FLAGS[CPPR][E->ien[CPPR][el].node[5]] & FBZ) {   // only check for the 5th node
          aid = 1;
	  get_global_1d_shape_fn(E,el,&GM,&dGamma,aid);
          }
      if (aid>=0)  {
        for(a=1;a<=onedvpts;a++)  {

	  for(j=1;j<=onedvpts;j++)  {
            dT[j] = 0.0;
	    for(k=1;k<=onedvpts;k++)
              dT[j] += E->M.vpt[GMVINDEX(k,j)]*BC[3][E->ien[CPPR][el].node[k+aid*onedvpts]];
            }
	  for(j=1;j<=onedvpts;j++)  {
	    Eres[a+aid*onedvpts] += dGamma.vpt[GMVGAMMA(aid,j)] *
		E->M.vpt[GMVINDEX(a,j)] * g_1d[j].weight[dims-1] *
		dT[j];
            }

	}
      }
    }

    return;
}


/* This function filters the temperature field. The temperature above   */
/* Tmax0(==1.0) and Tmin0(==0.0) is removed, while conserving the total */
/* energy. See Lenardic and Kaula, JGR, 1993.                           */
static void filter(struct All_variables *E)
{
    double Tsum0,Tmin,Tmax,Tsum1,TDIST,TDIST1;
    int m,i;
    double Tmax1,Tmin1;
    double *rhocp, sum_rhocp, total_sum_rhocp;
    int lev, nz;

    /* min and max temperature for filtering */
    const double Tmin0 = 0.0;
    const double Tmax0 = 1.0;

    Tsum0= Tsum1= 0.0;
    Tmin= Tmax= 0.0;
    Tmin1= Tmax1= 0.0;
    TDIST= TDIST1= 0.0;
    sum_rhocp = 0.0;

    lev=E->mesh.levmax;

    rhocp = (double *)malloc((E->lmesh.noz+1)*sizeof(double));
    for(i=1;i<=E->lmesh.noz;i++)
        rhocp[i] = E->refstate.rho[i] * E->refstate.heat_capacity[i];

    for(i=1;i<=E->lmesh.nno;i++)  {
        nz = ((i-1) % E->lmesh.noz) + 1;

        /* compute sum(rho*cp*T) before filtering, skipping nodes
           that's shared by another processor */
        if(!(E->NODE[lev][CPPR][i] & SKIP))
            Tsum0 += E->T[CPPR][i]*rhocp[nz];

        /* remove overshoot */
        if(E->T[CPPR][i]<Tmin)  Tmin=E->T[CPPR][i];
        if(E->T[CPPR][i]<Tmin0) E->T[CPPR][i]=Tmin0;
        if(E->T[CPPR][i]>Tmax) Tmax=E->T[CPPR][i];
        if(E->T[CPPR][i]>Tmax0) E->T[CPPR][i]=Tmax0;

    }

    /* find global max/min of temperature */
    MPI_Allreduce(&Tmin,&Tmin1,1,MPI_DOUBLE,MPI_MIN,E->parallel.world);
    MPI_Allreduce(&Tmax,&Tmax1,1,MPI_DOUBLE,MPI_MAX,E->parallel.world);

    for(i=1;i<=E->lmesh.nno;i++)  {
        nz = ((i-1) % E->lmesh.noz) + 1;

        /* remvoe undershoot */
        if(E->T[CPPR][i]<=fabs(2*Tmin0-Tmin1))   
          E->T[CPPR][i]=Tmin0;
        if(E->T[CPPR][i]>=(2*Tmax0-Tmax1))   
          E->T[CPPR][i]=Tmax0;

        /* sum(rho*cp*T) after filtering */
        if (!(E->NODE[lev][CPPR][i] & SKIP))  {
            Tsum1 += E->T[CPPR][i]*rhocp[nz];
            if(E->T[CPPR][i]!=Tmin0 && E->T[CPPR][i]!=Tmax0) {
                sum_rhocp += rhocp[nz];
            }
        }
    }

    /* find the difference of sum(rho*cp*T) before/after the filtering */
    TDIST=Tsum0-Tsum1;
    MPI_Allreduce(&TDIST,&TDIST1,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    MPI_Allreduce(&sum_rhocp,&total_sum_rhocp,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    TDIST=TDIST1/total_sum_rhocp;

    /* keep sum(rho*cp*T) the same before/after the filtering by distributing
       the difference back to nodes */
    for(i=1;i<=E->lmesh.nno;i++)   {
        if(E->T[CPPR][i]!=Tmin0 && E->T[CPPR][i]!=Tmax0)
            E->T[CPPR][i] +=TDIST;
    }

    free(rhocp);
}


static void process_visc_heating(struct All_variables *E, double *heating)
{
    void strain_rate_2_inv();
    int e, i;
    double visc, temp;
    float *strain_sqr;
    const int vpts = VPOINTS3D;

    strain_sqr = (float*) malloc((E->lmesh.nel+1)*sizeof(float));
    /* note that this will be negative for Atemp < 0 for time
       reversal */
    temp = E->control.disptn_number / E->control.Atemp / vpts;

    strain_rate_2_inv(E, strain_sqr, 0);

    for(e=1; e<=E->lmesh.nel; e++) {
        visc = 0.0;
        for(i = 1; i <= vpts; i++)
            visc += E->EVi[CPPR][(e-1)*vpts + i];

        heating[e] = temp * visc * strain_sqr[e];
    }

    free(strain_sqr);
}


static void process_adi_heating(struct All_variables *E, double *heating)
{
    int e, ez, i, j;
    double matprop, temp1, temp2;
    const int ends = ENODES3D;

    temp2 = E->control.disptn_number / ends;
    for(e=1; e<=E->lmesh.nel; e++) {
        ez = (e - 1) % E->lmesh.elz + 1;
        matprop = 0.125
            * (E->refstate.thermal_expansivity[ez] +
               E->refstate.thermal_expansivity[ez + 1])
            * (E->refstate.rho[ez] + E->refstate.rho[ez + 1])
            * (E->refstate.gravity[ez] + E->refstate.gravity[ez + 1]);

        temp1 = 0.0;
        for(i=1; i<=ends; i++) {
            j = E->ien[CPPR][e].node[i];
            temp1 += E->sphere.cap[CPPR].V[3][j]
                * (E->T[CPPR][j] + E->control.surface_temp);
        }

        heating[e] = matprop * temp1 * temp2;
    }
}


static void latent_heating(struct All_variables *E,
                           double *heating_latent, double *heating_adi,
                           float **B, float Ra, float clapeyron,
                           float depth, float transT, float inv_width)
{
    double temp, temp0, temp1, temp2, temp3, matprop;
    int e, ez, i, j;
    const int ends = ENODES3D;
    /* 
       note that this will be negative for time-reversal
    */
    temp0 = 2.0 * inv_width * clapeyron * E->control.disptn_number * Ra / E->control.Atemp / ends;
    temp1 = temp0 * clapeyron;

    for(e=1; e<=E->lmesh.nel; e++) {
        ez = (e - 1) % E->lmesh.elz + 1;
        matprop = 0.125
            * (E->refstate.thermal_expansivity[ez] +
               E->refstate.thermal_expansivity[ez + 1])
            * (E->refstate.rho[ez] + E->refstate.rho[ez + 1])
            * (E->refstate.gravity[ez] + E->refstate.gravity[ez + 1]);

        temp2 = 0;
        temp3 = 0;
        for(i=1; i<=ends; i++) {
            j = E->ien[CPPR][e].node[i];
            temp = (1.0 - B[CPPR][j]) * B[CPPR][j]
                * (E->T[CPPR][j] + E->control.surface_temp);
            temp2 += temp * E->sphere.cap[CPPR].V[3][j];
            temp3 += temp;
        }

        /* correction on the adiabatic cooling term */
        heating_adi[e] += matprop * temp2 * temp0;

        /* correction on the DT/Dt term */
        heating_latent[e] += temp3 * temp1;
    }
}


static void process_latent_heating(struct All_variables *E,
                                   double *heating_latent, double *heating_adi)
{
    int e;

    /* reset */
    for(e=1; e<=E->lmesh.nel; e++)
        heating_latent[e] = 1.0;

    if(E->control.Ra_410 != 0.0) {
        latent_heating(E, heating_latent, heating_adi,
                       E->Fas410, E->control.Ra_410,
                       E->control.clapeyron410, E->viscosity.z410,
                       E->control.transT410, E->control.inv_width410);

    }

    if(E->control.Ra_670 != 0.0) {
        latent_heating(E, heating_latent, heating_adi,
                       E->Fas670, E->control.Ra_670,
                       E->control.clapeyron670, E->viscosity.zlm,
                       E->control.transT670, E->control.inv_width670);
    }

    if(E->control.Ra_cmb != 0.0) {
        latent_heating(E, heating_latent, heating_adi,
                       E->Fascmb, E->control.Ra_cmb,
                       E->control.clapeyroncmb, E->viscosity.zcmb,
                       E->control.transTcmb, E->control.inv_widthcmb);
    }


    if(E->control.Ra_410 != 0 || E->control.Ra_670 != 0.0 ||
       E->control.Ra_cmb != 0) {
        for(e=1; e<=E->lmesh.nel; e++)
            heating_latent[e] = 1.0 / heating_latent[e];
    }
}


static double total_heating(struct All_variables *E, double **heating)
{
    int m, e;
    double sum, total;

    /* sum up within each processor */
    sum = 0;
    for(e=1; e<=E->lmesh.nel; e++)
        sum += heating[CPPR][e] * E->eco[CPPR][e].area;

    /* sum up for all processors */
    MPI_Allreduce(&sum, &total, 1, MPI_DOUBLE, MPI_SUM, E->parallel.world);

    return total;
}


static void process_heating(struct All_variables *E, int psc_pass)
{
    int m;
    double total_visc_heating, total_adi_heating;

    if(psc_pass == 0) {
        /* visc heating does not change between psc_pass, compute only
         * at first psc_pass */
        process_visc_heating(E, E->heating_visc[CPPR]);
    }
    process_adi_heating(E, E->heating_adi[CPPR]);
    process_latent_heating(E, E->heating_latent[CPPR], E->heating_adi[CPPR]);

    /* compute total amount of visc/adi heating over all processors
     * only at last psc_pass */
    if(psc_pass == (E->advection.temp_iterations-1)) {
        total_visc_heating = total_heating(E, E->heating_visc);
        total_adi_heating = total_heating(E, E->heating_adi);

        if(E->parallel.me == 0) {
            fprintf(E->fp, "Step: %d, Total_heating(visc, adi): %g %g\n",
                    E->monitor.solution_cycles,
                    total_visc_heating, total_adi_heating);
            fprintf(stderr, "Step: %d, Total_heating(visc, adi): %g %g\n",
                    E->monitor.solution_cycles,
                    total_visc_heating, total_adi_heating);
        }
    }
}
