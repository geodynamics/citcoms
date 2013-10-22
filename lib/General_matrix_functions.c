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
#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"

#ifdef _UNICOS
#include <fortran.h>
#endif

int epsilon[4][4] = {   /* Levi-Cita epsilon */
  {0, 0, 0, 0},
  {0, 1,-1, 1},
  {0,-1, 1,-1},
  {0, 1,-1, 1} };


/*  ===========================================================
    Iterative solver also using multigrid  ........
    ===========================================================  */
int solve_del2_u( struct All_variables *E, double *d0, double *F, double acc,
                  int high_lev )
{
  void assemble_del2_u();
  void e_assemble_del2_u();
  void n_assemble_del2_u();
  void strip_bcs_from_residual();
  void gauss_seidel();

  double conj_grad();
  double multi_grid();
  double global_vdot();
  void record();
  void report();

  int count,counts,cycles,convergent,valid;
  int i, neq, m;

  char message[200];

  double CPU_time0(),initial_time,time;
  double residual,prior_residual,r0;
  //double *D1[NCS], *r[NCS], *Au[NCS];

  neq  = E->lmesh.NEQ[high_lev];

  for(i=0;i<neq;i++)
    d0[i] = 0.0;

  r0=residual=sqrt(global_vdot(E,F,F,high_lev));

  prior_residual=2*residual;
  count = 0;
  initial_time=CPU_time0();

  if (!E->control.NMULTIGRID) {
    /* conjugate gradient solution */

    cycles = E->control.v_steps_low;
    residual = conj_grad(E,d0,F,acc,&cycles,high_lev);
    valid = (residual < acc)? 1:0;
  } else  {
    
    /* solve using multigrid  */

    counts =0;
    if(E->parallel.me==0){	/* output */
      snprintf(message,200,"resi = %.6e for iter %d acc %.6e",residual,counts,acc);
      record(E,message);
      report(E,message);
    }

    do {
      residual=multi_grid(E,d0,F,acc,high_lev);
      valid = (residual < acc)?1:0;
      counts ++;
      if(E->parallel.me==0){	/* output  */
	snprintf(message,200,"resi = %.6e for iter %d acc %.6e",residual,counts,acc);
	record(E,message);
	report(E,message);
      }
    }  while (!valid && counts < E->control.max_mg_cycles);

    cycles = counts;
  }


  /* Convergence check .....
     We should give it a chance to recover if it briefly diverges initially, and
     don't worry about slower convergence if it is close to the answer   */

  if(((count > 0) && (residual > r0*2.0))  ||
     ((fabs(residual-prior_residual) < acc*0.1) && (residual > acc * 10.0))   )
    convergent=0;
  else {
    convergent=1;
    prior_residual=residual;
  }

  if(E->control.print_convergence&&E->parallel.me==0)   {
    fprintf(E->fp,"%s residual (%03d) = %.3e from %.3e to %.3e in %5.2f secs \n",
	    (convergent ? " * ":"!!!"),cycles,residual,r0,acc,CPU_time0()-initial_time);
    fflush(E->fp);
  }

  count++;

  E->monitor.momentum_residual = residual;
  E->control.total_iteration_cycles += count;
  E->control.total_v_solver_calls += 1;

  return(valid);
}

/* =================================
   recursive multigrid function ....
   ================================= */
double multi_grid( struct All_variables *E, double *d1, double *F, double acc,
                   int hl /* higher level of two */ )
{
    double residual,AudotAu;
    void interp_vector();
    void project_vector();
    int m,i,j,Vn,Vnmax,cycles;
    double alpha,beta;
    void gauss_seidel();
    void element_gauss_seidel();
    void e_assemble_del2_u();
    void strip_bcs_from_residual();
    void n_assemble_del2_u();

    double conj_grad(),global_vdot();

    FILE *fp;
    char filename[1000];
    int lev,ic,ulev,dlev;

    const int levmin = E->mesh.levmin;
    const int levmax = E->mesh.levmax;

    double time1,time,CPU_time0();
    double *res[MAX_LEVELS],*AU[MAX_LEVELS];
    double *vel[MAX_LEVELS],*del_vel[MAX_LEVELS];
    double *rhs[MAX_LEVELS],*fl[MAX_LEVELS];

    /* because it's recursive, need a copy at each level */
    for(i=E->mesh.levmin;i<=E->mesh.levmax;i++){
      del_vel[i]=(double *)malloc((E->lmesh.NEQ[i]+1)*sizeof(double));
      AU[i] = (double *)malloc((E->lmesh.NEQ[i]+1)*sizeof(double));
      vel[i]=(double *)malloc((E->lmesh.NEQ[i]+1)*sizeof(double));
      res[i]=(double *)malloc((E->lmesh.NEQ[i])*sizeof(double));
      if (i<E->mesh.levmax)
        fl[i]=(double *)malloc((E->lmesh.NEQ[i])*sizeof(double));
    }

    Vnmax = E->control.mg_cycle;

    /* Project residual onto all the lower levels */

    project_vector(E,levmax,F,fl[levmax-1],1);
    strip_bcs_from_residual(E,fl[levmax-1],levmax-1);
    for(lev=levmax-1;lev>levmin;lev--) {
      project_vector(E,lev,fl[lev],fl[lev-1],1);
      strip_bcs_from_residual(E,fl[lev-1],lev-1);
    }

    /* Solve for the lowest level */

    /*    time=CPU_time0(); */
    cycles = E->control.v_steps_low;

    gauss_seidel(E,vel[levmin],fl[levmin],AU[levmin],acc*0.01,&cycles,levmin,0);

    for(lev=levmin+1;lev<=levmax;lev++) {
      time=CPU_time0();

      /* Utilize coarse solution and smooth at this level */
      interp_vector(E,lev-1,vel[lev-1],vel[lev]);
      strip_bcs_from_residual(E,vel[lev],lev);

      if (lev==levmax)
        for(j=0;j<E->lmesh.NEQ[lev];j++)
           res[lev][j]=F[j];
      else
        for(j=0;j<E->lmesh.NEQ[lev];j++)
           res[lev][j]=fl[lev][j];

         for(Vn=1;Vn<=Vnmax;Vn++) {
          /*    Downward stoke of the V    */
           for (dlev=lev;dlev>=levmin+1;dlev--)   {

            /* Pre-smoothing  */
            cycles=((dlev==levmax)?
                E->control.v_steps_high:E->control.down_heavy);
            ic = ((dlev==lev)?1:0);
            gauss_seidel(E,vel[dlev],res[dlev],AU[dlev],0.01,&cycles,dlev,ic);

            for(i=0;i<E->lmesh.NEQ[dlev];i++)
              res[dlev][i]  = res[dlev][i] - AU[dlev][i];

            project_vector(E,dlev,res[dlev],res[dlev-1],1);
            strip_bcs_from_residual(E,res[dlev-1],dlev-1);
           }

           /*    Bottom of the V    */
           cycles = E->control.v_steps_low;
           gauss_seidel(E,vel[levmin],res[levmin],AU[levmin],acc*0.01,&cycles,
               levmin,0);
           /*    Upward stoke of the V    */
           for (ulev=levmin+1;ulev<=lev;ulev++)   {
              cycles=((ulev==levmax)?
                  E->control.v_steps_high:E->control.up_heavy);

              interp_vector(E,ulev-1,vel[ulev-1],del_vel[ulev]);
              strip_bcs_from_residual(E,del_vel[ulev],ulev);
              gauss_seidel(E,del_vel[ulev],res[ulev],AU[ulev],0.01,&cycles,
                  ulev,1);

              AudotAu = global_vdot(E,AU[ulev],AU[ulev],ulev);
              alpha = global_vdot(E,AU[ulev],res[ulev],ulev)/AudotAu;

              for(i=0;i<E->lmesh.NEQ[ulev];i++)   {
                vel[ulev][i] += alpha*del_vel[ulev][i];
              }

              if (ulev ==levmax)
                for(i=0;i<E->lmesh.NEQ[ulev];i++) {
                  res[ulev][i] -= alpha*AU[ulev][i];
                }

            }
          }
      }

    for(j=0;j<E->lmesh.NEQ[levmax];j++)   {
      F[j]=res[levmax][j];
      d1[j]+=vel[levmax][j];
    }

    residual = sqrt(global_vdot(E,F,F,hl));

    for(i=E->mesh.levmin;i<=E->mesh.levmax;i++) {
      free((double*) del_vel[i]);
      free((double*) AU[i]);
      free((double*) vel[i]);
      free((double*) res[i]);
      if (i<E->mesh.levmax)
        free((double*) fl[i]);
	  }

    return(residual);
}

/*  ===========================================================
    Conjugate gradient relaxation for the matrix equation Kd = f
    Returns the residual reduction after itn iterations ...
    ===========================================================  */
#ifndef USE_CUDA
double conj_grad( struct All_variables *E,
                  double *d0, double *F, double acc,
                  int *cycles, int level )
{
    double *r0,*r1,*r2;
    double *z0,*z1,*z2;
    double *p1,*p2;
    double *Ap;
    double *shuffle;

    int m,count,i,steps;
    double residual;
    double alpha,beta,dotprod,dotr1z1,dotr0z0;

    double CPU_time0(),time;

    void parallel_process_termination();
    void assemble_del2_u();
    void strip_bcs_from_residual();
    double global_vdot();

    const int mem_lev=E->mesh.levmax;
    const int high_neq = E->lmesh.NEQ[level];

    steps = *cycles;

    r0 = (double *)malloc(E->lmesh.NEQ[mem_lev]*sizeof(double));
    r1 = (double *)malloc(E->lmesh.NEQ[mem_lev]*sizeof(double));
    r2 = (double *)malloc(E->lmesh.NEQ[mem_lev]*sizeof(double));
    z0 = (double *)malloc(E->lmesh.NEQ[mem_lev]*sizeof(double));
    z1 = (double *)malloc(E->lmesh.NEQ[mem_lev]*sizeof(double));
    p1 = (double *)malloc((1+E->lmesh.NEQ[mem_lev])*sizeof(double));
    p2 = (double *)malloc((1+E->lmesh.NEQ[mem_lev])*sizeof(double));
    Ap = (double *)malloc((1+E->lmesh.NEQ[mem_lev])*sizeof(double));

    for(i=0;i<high_neq;i++) {
      r1[i] = F[i];
      d0[i] = 0.0;
    }

    residual = sqrt(global_vdot(E,r1,r1,level));

    assert(residual != 0.0  /* initial residual for CG = 0.0 */);
    count = 0;

    while (((residual > acc) && (count < steps)) || count == 0)  {

      for(i=0;i<high_neq;i++)
        z1[i] = E->BI[level][i] * r1[i];

      dotr1z1 = global_vdot(E,r1,z1,level);

      if (count == 0 )
        for(i=0;i<high_neq;i++)
          p2[i] = z1[i];
      else {
        assert(dotr0z0 != 0.0 /* in head of conj_grad */);
        beta = dotr1z1/dotr0z0;
          for(i=0;i<high_neq;i++)
            p2[i] = z1[i] + beta * p1[i];
      }

      dotr0z0 = dotr1z1;

      assemble_del2_u(E,p2,Ap,level,1);

      dotprod=global_vdot(E,p2,Ap,level);

      if(0.0==dotprod)
        alpha=1.0e-3;
      else
        alpha = dotr1z1/dotprod;

      for(i=0;i<high_neq;i++) {
        d0[i] += alpha * p2[i];
        r2[i] = r1[i] - alpha * Ap[i];
	    }

      residual = sqrt(global_vdot(E,r2,r2,level));

      shuffle = r0; r0 = r1; r1 = r2; r2 = shuffle;
      shuffle = z0; z0 = z1; z1 = shuffle;
      shuffle = p1; p1 = p2; p2 = shuffle;

      count++;

    } /* end of while-loop */

    *cycles=count;

    strip_bcs_from_residual(E,d0,level);

    free((double*) r0);
    free((double*) r1);
    free((double*) r2);
    free((double*) z0);
    free((double*) z1);
    free((double*) p1);
    free((double*) p2);
    free((double*) Ap);

    return(residual);   }
#endif /* !USE_CUDA */


/* =============================================================================
An element by element version of the gauss-seidel routine. Initially this is a 
test platform, we want to know if it handles discontinuities any better than 
the node/equation versions
==============================================================================*/
void element_gauss_seidel( struct All_variables *E,
                           double *d0, double *F, double *Ad,
                           double acc, int *cycles, int level, int guess )
{
    int count,i,j,k,l,m,ns,nc,d,steps,loc;
    int p1,p2,p3,q1,q2,q3;
    int e,eq,node,node1;
    int element,eqn1,eqn2,eqn3,eqn11,eqn12,eqn13;

    void e_assemble_del2_u();
    void n_assemble_del2_u();
    void strip_bcs_from_residual();

    double U1[24],AD1[24],F1[24];
    double w1,w2,w3;
    double w11,w12,w13;
    double w[24];

    double *dd;
    double *elt_k;
    int *vis;

    const int dims=E->mesh.nsd;
    const int ends=enodes[dims];
    const int n=loc_mat_size[E->mesh.nsd];
    const int neq=E->lmesh.NEQ[level];
    const int nel=E->lmesh.NEL[level];
    const int nno=E->lmesh.NNO[level];


    steps=*cycles;

    dd = (double *)malloc(neq*sizeof(double));
    vis = (int *)malloc((nno+1)*sizeof(int));
    elt_k=(double *)malloc((24*24)*sizeof(double));

    if(guess){
      e_assemble_del2_u(E,d0,Ad,level,1);
    }
    else {
      for(i=0;i<neq;i++)
	     Ad[i]=d0[i]=0.0;
    }

   count=0;
   while (count <= steps) {
    for(i=1;i<=nno;i++)
      vis[i]=0;

    for(e=1;e<=nel;e++) {
      elt_k = E->elt_k[level][e].k;

      for(i=1;i<=ends;i++)  {
        node=E->IEN[level][e].node[i];
        p1=(i-1)*dims;
        w[p1] = w[p1+1] = w[p1+2] = 1.0;
        if(E->NODE[level][node] & VBX)
          w[p1] = 0.0;
        if(E->NODE[level][node] & VBY)
          w[p1+1] = 0.0;
        if(E->NODE[level][node] & VBZ)
          w[p1+2] = 0.0;
      }

      for(i=1;i<=ends;i++) {
        node=E->IEN[level][e].node[i];
        if(!vis[node])
          continue;

        eqn1=E->ID[level][node].doff[1];
        eqn2=E->ID[level][node].doff[2];
        eqn3=E->ID[level][node].doff[3];
        p1=(i-1)*dims*n;
        p2=p1+n;
        p3=p2+n;

        /* update Au */
        for(j=1;j<=ends;j++) {
          node1=E->IEN[level][e].node[j];

          eqn11=E->ID[level][node1].doff[1];
          eqn12=E->ID[level][node1].doff[2];
          eqn13=E->ID[level][node1].doff[3];
          q1=(j-1)*3;

          Ad[eqn11] += w[q1]*(elt_k[p1+q1]*dd[eqn1] + elt_k[p2+q1]*dd[eqn2] 
              + elt_k[p3+q1]*dd[eqn3]);
          Ad[eqn12] += w[q1+1]*(elt_k[p1+q1+1]*dd[eqn1] + 
              elt_k[p2+q1+1]*dd[eqn2] + elt_k[p3+q1+1]*dd[eqn3]);
          Ad[eqn13] += w[q1+2]*(elt_k[p1+q1+2]*dd[eqn1] + 
              elt_k[p2+q1+2]*dd[eqn2] + elt_k[p3+q1+2] * dd[eqn3]);
        }
      }


      for(i=1;i<=ends;i++) {
        node=E->IEN[level][e].node[i];
        if(vis[node])
          continue;

        eqn1=E->ID[level][node].doff[1];
        eqn2=E->ID[level][node].doff[2];
        eqn3=E->ID[level][node].doff[3];
        p1=(i-1)*dims*n;
        p2=p1+n;
        p3=p2+n;

        /* update dd, d0 */
        d0[eqn1] += (dd[eqn1] = 
            w[(i-1)*dims+0]*(F[eqn1]-Ad[eqn1])*E->BI[level][eqn1]);
        d0[eqn2] += (dd[eqn2] = 
            w[(i-1)*dims+1]*(F[eqn2]-Ad[eqn2])*E->BI[level][eqn2]);
        d0[eqn3] += (dd[eqn3] = 
            w[(i-1)*dims+2]*(F[eqn3]-Ad[eqn3])*E->BI[level][eqn3]);

        vis[node]=1;

        /* update Au */
        for(j=1;j<=ends;j++) {
          node1=E->IEN[level][e].node[j];

        eqn11=E->ID[level][node1].doff[1];
        eqn12=E->ID[level][node1].doff[2];
        eqn13=E->ID[level][node1].doff[3];
        q1=(j-1)*3;
        q2=q1+1;
        q3=q1+2;

        Ad[eqn11] += w[q1]*(elt_k[p1+q1] * dd[eqn1] + 
            elt_k[p2+q1] * dd[eqn2] + elt_k[p3+q1] * dd[eqn3]);
        Ad[eqn12] += w[q2]*(elt_k[p1+q2] * dd[eqn1] + 
            elt_k[p2+q2] * dd[eqn2] + elt_k[p3+q2] * dd[eqn3]);
        Ad[eqn13] += w[q3]*(elt_k[p1+q3] * dd[eqn1] + 
            elt_k[p2+q3] * dd[eqn2] + elt_k[p3+q3] * dd[eqn3]);
        }
      }

    }          /* end for el  */

        (E->solver.exchange_id_d)(E, Ad, level);
        (E->solver.exchange_id_d)(E, d0, level);

	/* completed cycle */

	    count++;

    }

    free((double*) dd);
    free((int*) vis);
    free((double*) elt_k);
}


#ifndef USE_CUDA

/* ============================================================================
 Multigrid Gauss-Seidel relaxation scheme which requires the storage of local
 information, otherwise some other method is required. NOTE this is a bit worse
 than real gauss-seidel because it relaxes all the equations for a node at one
 time (Jacobi at a node). It does the job though.
 ============================================================================ */
void gauss_seidel( struct All_variables *E,
                   double *d0, double *F, double *Ad, double acc,
                   int *cycles, int level, int guess )
{
    int count,i,j,k,l,m,ns,steps;
    int *C;
    int eqn1,eqn2,eqn3;

    void parallel_process_termination();
    void n_assemble_del2_u();

    double U1,U2,U3,UU;
    double sor,residual,global_vdot();

    higher_precision *B1,*B2,*B3;


    const int dims=E->mesh.nsd;
    const int ends=enodes[dims];
    const int n=loc_mat_size[E->mesh.nsd];
    const int neq=E->lmesh.NEQ[level];
    const int num_nodes=E->lmesh.NNO[level];
    const int nox=E->lmesh.NOX[level];
    const int noz=E->lmesh.NOY[level];
    const int noy=E->lmesh.NOZ[level];
    const int max_eqn=14*dims;

    const double zeroo = 0.0;

    steps=*cycles;
    sor = 1.3;

    if(guess)
      n_assemble_del2_u(E,d0,Ad,level,1);
    else
      for(i=0;i<neq;i++)
        d0[i]=Ad[i]=zeroo;

    count = 0;


    while (count < steps) {
      for(j=0;j<=E->lmesh.NEQ[level];j++)
        E->temp[j] = zeroo;

      Ad[neq] = zeroo;

      for(i=1;i<=E->lmesh.NNO[level];i++)
        if(E->NODE[level][i] & OFFSIDE)   {
          eqn1=E->ID[level][i].doff[1];
          eqn2=E->ID[level][i].doff[2];
          eqn3=E->ID[level][i].doff[3];
          E->temp[eqn1] = (F[eqn1] - Ad[eqn1])*E->BI[level][eqn1];
          E->temp[eqn2] = (F[eqn2] - Ad[eqn2])*E->BI[level][eqn2];
          E->temp[eqn3] = (F[eqn3] - Ad[eqn3])*E->BI[level][eqn3];
          E->temp1[eqn1] = Ad[eqn1];
          E->temp1[eqn2] = Ad[eqn2];
          E->temp1[eqn3] = Ad[eqn3];
        }

      for(i=1;i<=E->lmesh.NNO[level];i++) {
        eqn1=E->ID[level][i].doff[1];
        eqn2=E->ID[level][i].doff[2];
        eqn3=E->ID[level][i].doff[3];
        C=E->Node_map[level]+(i-1)*max_eqn;
        B1=E->Eqn_k1[level]+(i-1)*max_eqn;
        B2=E->Eqn_k2[level]+(i-1)*max_eqn;
        B3=E->Eqn_k3[level]+(i-1)*max_eqn;

       /* Ad on boundaries differs after the following operation, but
        no communications are needed yet, because boundary Ad will
        not be used for the G-S iterations for interior nodes */

        for(j=3;j<max_eqn;j++) {
           UU = E->temp[C[j]];
           Ad[eqn1] += B1[j]*UU;
           Ad[eqn2] += B2[j]*UU;
           Ad[eqn3] += B3[j]*UU;
        }

        if(!(E->NODE[level][i]&OFFSIDE)) {
           E->temp[eqn1] = (F[eqn1] - Ad[eqn1])*E->BI[level][eqn1];
           E->temp[eqn2] = (F[eqn2] - Ad[eqn2])*E->BI[level][eqn2];
           E->temp[eqn3] = (F[eqn3] - Ad[eqn3])*E->BI[level][eqn3];
	      }

        /* Ad on boundaries differs after the following operation */
        for(j=0;j<max_eqn;j++)
		      Ad[C[j]] += B1[j]*E->temp[eqn1]
                   +  B2[j]*E->temp[eqn2]
                   +  B3[j]*E->temp[eqn3];

        d0[eqn1] += E->temp[eqn1];
        d0[eqn2] += E->temp[eqn2];
        d0[eqn3] += E->temp[eqn3];
      }

      for(i=1;i<=E->lmesh.NNO[level];i++)
        if(E->NODE[level][i] & OFFSIDE)   {
          eqn1=E->ID[level][i].doff[1];
          eqn2=E->ID[level][i].doff[2];
          eqn3=E->ID[level][i].doff[3];
          Ad[eqn1] -= E->temp1[eqn1];
          Ad[eqn2] -= E->temp1[eqn2];
          Ad[eqn3] -= E->temp1[eqn3];
        }

      (E->solver.exchange_id_d)(E, Ad, level);

      for(i=1;i<=E->lmesh.NNO[level];i++)
        if(E->NODE[level][i] & OFFSIDE)   {
          eqn1=E->ID[level][i].doff[1];
          eqn2=E->ID[level][i].doff[2];
          eqn3=E->ID[level][i].doff[3];
          Ad[eqn1] += E->temp1[eqn1];
          Ad[eqn2] += E->temp1[eqn2];
          Ad[eqn3] += E->temp1[eqn3];
        }


      count++;
    }
    *cycles=count;
}
#endif /* !USE_CUDA */

/* Fast (conditional) determinant for 3x3 or 2x2 ... otherwise calls general routine */

double determinant( double A[4][4], int n )
{ 
  double gen_determinant();

  switch(n) { 
    case 1:
      return(A[1][1]);
    case 2:
      return(A[1][1]*A[2][2]-A[1][2]*A[2][1]);
    case 3:
      return(A[1][1]*(A[2][2]*A[3][3]-A[2][3]*A[3][2])-
             A[1][2]*(A[2][1]*A[3][3]-A[2][3]*A[3][1])+
             A[1][3]*(A[2][1]*A[3][2]-A[2][2]*A[3][1]));
    default:
      return(1);
  }
}



double cofactor(A,i,j,n)
     double A[4][4];
     int i,j,n;

{ int k,l,p,q;
  double determinant();

  double B[4][4]; /* because of recursive behaviour of det/cofac, need to use
			       new copy of B at each 'n' level of this routine */

  if (n>3) printf("Error, no cofactors for matrix more than 3x3\n");

  p=q=1;

  for(k=1;k<=n;k++)    {
     if(k==i) continue;
     for(l=1;l<=n;l++)      {
	   if (l==j) continue;
           B[p][q]=A[k][l];
	   q++ ;
	   }
     q=1;p++;
     }


  return(epsilon[i][j]*determinant(B,n-1));


}

long double lg_pow(long double a, int n)
{
    /* compute the value of "a" raised to the power of "n" */
    long double b = 1.0;
    int i;

    for(i=0; i<n; i++) {
        b = b*a;
    }

    return(b);
}


