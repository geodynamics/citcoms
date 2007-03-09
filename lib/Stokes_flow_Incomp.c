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
/*   Functions which solve for the velocity and pressure fields using Uzawa-type iteration loop.  */

#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include <stdlib.h>

extern int Emergency_stop;

/* Master loop for pressure and (hence) velocity field */


void solve_constrained_flow_iterative(E)
     struct All_variables *E;

{
    double *D1;
    double *u;
    double *R,*Bp;
    double residual_ddash;
    double vmag;
    double global_vdot(),global_pdot();

    float solve_Ahat_p_fhat();
    void assemble_del2_u();
    void assemble_grad_p();
    void assemble_div_u();
    void v_from_vector();
    void p_to_nodes();
    void strip_bcs_from_residual();
    void velocities_conform_bcs();

    int steps,cycles;
    int i,j,k,doff,vel_cycles_previous,vel_calls_previous;

    double time,CPU_time0();

    const int npno = E->lmesh.npno;
    const int gnpno = E->mesh.npno;
    const int nno = E->lmesh.nno;
    const int dims = E->mesh.nsd;
    const int neq = E->lmesh.neq;
    const int gneq = E->mesh.neq;
    const int addi_dof = additional_dof[dims];

    time=CPU_time0();

    cycles=E->control.p_iterations;

    /* Solve for velocity and pressure, correct for bc's */

    residual_ddash=solve_Ahat_p_fhat(E,E->U,E->P,E->F,E->control.accuracy,&cycles);

    v_from_vector(E);
    p_to_nodes(E,E->P,E->NP,E->mesh.levmax);

/* */

  return;
}

void solve_constrained_flow_iterative_pseudo_surf(E)
     struct All_variables *E;

{
    double *D1;
    double *u;
    double *R,*Bp;
    double residual_ddash;
    double vmag;
    double global_vdot(),global_pdot();

    float solve_Ahat_p_fhat();
    void v_from_vector_pseudo_surf();
    void p_to_nodes();

    int steps,cycles;
    int i,j,k,doff,vel_cycles_previous,vel_calls_previous;

    double time,CPU_time0();

    const int npno = E->lmesh.npno;
    const int gnpno = E->mesh.npno;
    const int nno = E->lmesh.nno;
    const int dims = E->mesh.nsd;
    const int neq = E->lmesh.neq;
    const int gneq = E->mesh.neq;
    const int addi_dof = additional_dof[dims];

    time=CPU_time0();

    cycles=E->control.p_iterations;

    /* Solve for velocity and pressure, correct for bc's */

    residual_ddash=solve_Ahat_p_fhat(E,E->U,E->P,E->F,E->control.accuracy,&cycles);

    v_from_vector_pseudo_surf(E);
    p_to_nodes(E,E->P,E->NP,E->mesh.levmax);

/* */

  return;
}


/*  ==========================================================================  */

float solve_Ahat_p_fhat(E,V,P,F,imp,steps_max)

     struct All_variables *E;
     double **V,**P,**F;
     double imp;
     int *steps_max;

{
  int m,i,j,k,ii,count,convergent,valid,problems,lev,lev_low,npno,neq,steps;
  int gnpno,gneq;

  double *r1[NCS],*R[NCS];
  double *r0[NCS],*r2[NCS],*z0[NCS],*z1[NCS],*s1[NCS],*s2[NCS],*Ah[NCS];
  double *shuffle[NCS];
  double alpha,delta,s2dotAhat,r0dotr0,r1dotz1;
  double residual, initial_residual, last_residual,v_res;

  double global_vdot(),global_pdot();
  double *dvector();

  double time0,time,CPU_time0();
  float dpressure,dvelocity;

  void assemble_div_u();
  void assemble_del2_u();
  void assemble_grad_p();
  void strip_bcs_from_residual();
  int  solve_del2_u();
  void parallel_process_termination();

  const int dims=E->mesh.nsd;
  const int n=loc_mat_size[E->mesh.nsd];

  gnpno=E->mesh.npno;
  gneq=E->mesh.neq;

  for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    npno=E->lmesh.npno;
    neq=E->lmesh.neq;
    r0[m] = (double *)malloc((npno+1)*sizeof(double));
    r1[m] = (double *)malloc((npno+1)*sizeof(double));
    r2[m] = (double *)malloc((npno+1)*sizeof(double));
    z0[m] = (double *)malloc((npno+1)*sizeof(double));
    z1[m] = (double *)malloc((npno+1)*sizeof(double));
    s1[m] = (double *)malloc((npno+1)*sizeof(double));
    s2[m] = (double *)malloc((npno+1)*sizeof(double));
    }


  problems=0;
  time0=time=CPU_time0();

  /* calculate the velocity residual, note there are tricks involved here */

  lev=E->mesh.levmax;


  v_res=sqrt(global_vdot(E,F,F,lev)/gneq);


  if (E->parallel.me==0)  {
    fprintf(E->fp,"initial residue of momentum equation F %.9e %d\n",v_res,gneq);
    fprintf(stderr,"initial residue of momentum equation F %.9e %d\n",v_res,gneq);
  }


  assemble_grad_p(E,P,E->u1,lev);


  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=0;i<neq;i++)
      F[m][i] = F[m][i] - E->u1[m][i];

  assemble_del2_u(E,V,E->u1,lev,1);

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=0;i<neq;i++)
      F[m][i] = F[m][i] - E->u1[m][i];

  strip_bcs_from_residual(E,F,lev);

  valid=solve_del2_u(E,E->u1,F,imp*v_res,E->mesh.levmax);
  strip_bcs_from_residual(E,E->u1,lev);

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=0;i<neq;i++)
      V[m][i] += E->u1[m][i];

  assemble_div_u(E,V,r1,lev);

  residual = initial_residual = sqrt(global_pdot(E,r1,r1,lev)/gnpno);

  E->monitor.vdotv = sqrt(global_vdot(E,V,V,lev)/gneq);

  E->monitor.incompressibility = residual/E->monitor.vdotv;

  count = 0;
  convergent=0;

  if (E->control.print_convergence && E->parallel.me==0)  {
    fprintf(E->fp,"AhatP (%03d) after %g seconds with div/v=%.3e for step %d\n",count,CPU_time0()-time0,E->monitor.incompressibility,E->monitor.solution_cycles); /**/
    fflush(E->fp);
    fprintf(stderr,"AhatP (%03d) after %g seconds with div/v=%.3e for step %d\n",count,CPU_time0()-time0,E->monitor.incompressibility,E->monitor.solution_cycles); /**/
  }

  dpressure = 1.0;
  dvelocity = 1.0;

  while( (valid) && (count < *steps_max) &&
	 (E->monitor.incompressibility >= E->control.tole_comp) &&
	 (dpressure >= imp) && (dvelocity >= imp) )  {

    for (m=1;m<=E->sphere.caps_per_proc;m++)
      for(j=1;j<=npno;j++)
	z1[m][j] = E->BPI[lev][m][j]*r1[m][j];

    r1dotz1 = global_pdot(E,r1,z1,lev);

    if ((count == 0))
      for (m=1;m<=E->sphere.caps_per_proc;m++)
	for(j=1;j<=npno;j++)
	  s2[m][j] = z1[m][j];
    else {
      r0dotr0=global_pdot(E,r0,z0,lev);
      assert(r0dotr0 != 0.0  /* Division by zero in head of incompressibility iteration */);
      delta = r1dotz1/r0dotr0;
      for (m=1;m<=E->sphere.caps_per_proc;m++)
	for(j=1;j<=npno;j++)
	  s2[m][j] = z1[m][j] + delta * s1[m][j];
    }

    assemble_grad_p(E,s2,F,lev);

    valid=solve_del2_u(E,E->u1,F,imp*v_res,lev);
    strip_bcs_from_residual(E,E->u1,lev);

    assemble_div_u(E,E->u1,F,lev);

    s2dotAhat=global_pdot(E,s2,F,lev);

    if (valid)
      /* alpha defined this way is the same as R&W */
      alpha = r1dotz1/s2dotAhat;
    else
      alpha = 0.0;

    for (m=1;m<=E->sphere.caps_per_proc;m++)
      for(j=1;j<=npno;j++)   {
	r2[m][j] = r1[m][j] - alpha * F[m][j];
	P[m][j] += alpha * s2[m][j];
      }

    for (m=1;m<=E->sphere.caps_per_proc;m++)
      for(j=0;j<neq;j++)
	V[m][j] -= alpha * E->u1[m][j];


    assemble_div_u(E,V,F,lev);
    E->monitor.vdotv = global_vdot(E,V,V,E->mesh.levmax);
    E->monitor.incompressibility = sqrt((gneq/gnpno)*(1.0e-32+global_pdot(E,F,F,lev)/(1.0e-32+E->monitor.vdotv)));
    dpressure = alpha * sqrt(global_pdot(E,s2,s2,lev)
			     / (1.0e-32 + global_pdot(E,P,P,lev)));
    dvelocity = alpha * sqrt(global_vdot(E,E->u1,E->u1,lev)
			     / (1.0e-32 + E->monitor.vdotv));

    count++;
    if (E->control.print_convergence && E->parallel.me==0)  {
      fprintf(E->fp, "AhatP (%03d) after %g seconds with div/v=%.3e dv/v=%.3e"
	      " and dp/p=%.3e for step %d\n",
	      count, CPU_time0()-time0, E->monitor.incompressibility,
	      dvelocity, dpressure, E->monitor.solution_cycles);
      fprintf(stderr, "AhatP (%03d) after %g seconds with div/v=%.3e dv/v=%.3e"
	      " and dp/p=%.3e for step %d\n",
	      count, CPU_time0()-time0, E->monitor.incompressibility,
	      dvelocity, dpressure, E->monitor.solution_cycles);
    }

    for (m=1;m<=E->sphere.caps_per_proc;m++)    {
      shuffle[m]=s1[m];s1[m]=s2[m];s2[m]=shuffle[m];
      shuffle[m]=r0[m];r0[m]=r1[m];r1[m]=r2[m];r2[m]=shuffle[m];
      shuffle[m]=z0[m];z0[m]=z1[m];z1[m]=shuffle[m];
    }

  }       /* end loop for conjugate gradient   */

  if(problems) {
    fprintf(E->fp,"Convergence of velocity solver may affect continuity\n");
    fprintf(E->fp,"Consider running with the `see_convergence=on' option\n");
    fprintf(E->fp,"To evaluate the performance of the current relaxation parameters\n");
    fflush(E->fp);
  }

  for (m=1;m<=E->sphere.caps_per_proc;m++)    {
    free((void *) r0[m]);
    free((void *) r1[m]);
    free((void *) r2[m]);
    free((void *) z0[m]);
    free((void *) z1[m]);
    free((void *) s1[m]);
    free((void *) s2[m]);
  }

  *steps_max=count;

  return(residual);
}

/*  ==========================================================================  */




void v_from_vector(E)
	struct All_variables *E;
{
	int m,node;
	const int nno = E->lmesh.nno;

	for (m=1;m<=E->sphere.caps_per_proc;m++)   {
		for(node=1;node<=nno;node++)     {
			E->sphere.cap[m].V[1][node] = E->U[m][E->id[m][node].doff[1]];
			E->sphere.cap[m].V[2][node] = E->U[m][E->id[m][node].doff[2]];
			E->sphere.cap[m].V[3][node] = E->U[m][E->id[m][node].doff[3]];
			if (E->node[m][node] & VBX)
				E->sphere.cap[m].V[1][node] = E->sphere.cap[m].VB[1][node];
			if (E->node[m][node] & VBY)
				E->sphere.cap[m].V[2][node] = E->sphere.cap[m].VB[2][node];
			if (E->node[m][node] & VBZ)
				E->sphere.cap[m].V[3][node] = E->sphere.cap[m].VB[3][node];
		}
	}

	return;
}

void v_from_vector_pseudo_surf(E)
	struct All_variables *E;
{
	int m,node;

	const int nno = E->lmesh.nno;
	const int dofs = E->mesh.dof;
	double sum_V = 0.0, sum_dV = 0.0, rel_error = 0.0, global_max_error = 0.0;
	double tol_error = 1.0e-03;

	for (m=1;m<=E->sphere.caps_per_proc;m++)   {
		for(node=1;node<=nno;node++)     {
			E->sphere.cap[m].Vprev[1][node] = E->sphere.cap[m].V[1][node];
			E->sphere.cap[m].Vprev[2][node] = E->sphere.cap[m].V[2][node];
			E->sphere.cap[m].Vprev[3][node] = E->sphere.cap[m].V[3][node];

			E->sphere.cap[m].V[1][node] = E->U[m][E->id[m][node].doff[1]];
			E->sphere.cap[m].V[2][node] = E->U[m][E->id[m][node].doff[2]];
			E->sphere.cap[m].V[3][node] = E->U[m][E->id[m][node].doff[3]];
			if (E->node[m][node] & VBX)
				E->sphere.cap[m].V[1][node] = E->sphere.cap[m].VB[1][node];
			if (E->node[m][node] & VBY)
				E->sphere.cap[m].V[2][node] = E->sphere.cap[m].VB[2][node];
			if (E->node[m][node] & VBZ)
				E->sphere.cap[m].V[3][node] = E->sphere.cap[m].VB[3][node];

			sum_dV += (E->sphere.cap[m].V[1][node] - E->sphere.cap[m].Vprev[1][node])*(E->sphere.cap[m].V[1][node] - E->sphere.cap[m].Vprev[1][node])
				+ (E->sphere.cap[m].V[2][node] - E->sphere.cap[m].Vprev[2][node])*(E->sphere.cap[m].V[2][node] - E->sphere.cap[m].Vprev[2][node])
				+ (E->sphere.cap[m].V[3][node] - E->sphere.cap[m].Vprev[3][node])*(E->sphere.cap[m].V[3][node] - E->sphere.cap[m].Vprev[3][node]);
			sum_V += E->sphere.cap[m].V[1][node]*E->sphere.cap[m].V[1][node]
				+ E->sphere.cap[m].V[2][node]*E->sphere.cap[m].V[2][node]
				+ E->sphere.cap[m].V[3][node]*E->sphere.cap[m].V[3][node];
		}
		rel_error = sqrt(sum_dV)/sqrt(sum_V);
		MPI_Allreduce(&rel_error,&global_max_error,1,MPI_DOUBLE,MPI_MAX,E->parallel.world);
		if(global_max_error <= tol_error) E->monitor.stop_topo_loop = 1;
		if(E->parallel.me==0)
			fprintf(stderr,"global_max_error=%e stop_topo_loop=%d\n",global_max_error,E->monitor.stop_topo_loop);

	}

	return;
}

void velo_from_element(E,VV,m,el,sphere_key)
  struct All_variables *E;
  float VV[4][9];
  int el,m,sphere_key;
  {

  int a, node;
  double sint, cost, sinf, cosf;
  const int dims=E->mesh.nsd;
  const int ends=enodes[E->mesh.nsd];
  const int nno=E->lmesh.nno;
  const int lev=E->mesh.levmax;

  if (sphere_key)
    for(a=1;a<=ends;a++)   {
      node = E->ien[m][el].node[a];
      VV[1][a] = E->sphere.cap[m].V[1][node];
      VV[2][a] = E->sphere.cap[m].V[2][node];
      VV[3][a] = E->sphere.cap[m].V[3][node];
      }
  else {
    for(a=1;a<=ends;a++)   {
      node = E->ien[m][el].node[a];

      sint = E->SinCos[lev][m][0][node];
      sinf = E->SinCos[lev][m][1][node];
      cost = E->SinCos[lev][m][2][node];
      cosf = E->SinCos[lev][m][3][node];

      VV[1][a] = E->sphere.cap[m].V[1][node]*cost*cosf
          - E->sphere.cap[m].V[2][node]*sinf
          + E->sphere.cap[m].V[3][node]*sint*cosf;
      VV[2][a] = E->sphere.cap[m].V[1][node]*cost*sinf
          + E->sphere.cap[m].V[2][node]*cosf
          + E->sphere.cap[m].V[3][node]*sint*sinf;
      VV[3][a] = -E->sphere.cap[m].V[1][node]*sint
          + E->sphere.cap[m].V[3][node]*cost;
      }
  }
  return;
  }
