/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */


/*   Functions which solve for the velocity and pressure fields using Uzawa-type iteration loop.  */

#include <math.h>
#include <string.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include <stdlib.h>

void myerror(struct All_variables *,char *);

static void solve_Ahat_p_fhat(struct All_variables *E,
                              double *V, double *P, double *F,
                              double imp, int *steps_max);
static void solve_Ahat_p_fhat_CG(struct All_variables *E,
                                 double *V, double *P, double *F,
                                 double imp, int *steps_max);
static void solve_Ahat_p_fhat_BiCG(struct All_variables *E,
                                    double *V, double *P, double *F,
                                    double imp, int *steps_max);
static void solve_Ahat_p_fhat_iterCG(struct All_variables *E,
                                      double *V, double *P, double *F,
                                      double imp, int *steps_max);
static void initial_vel_residual(struct All_variables *E,
                                 double *V, double *P, double *F,
                                 double imp);


/* Master loop for pressure and (hence) velocity field */

void solve_constrained_flow_iterative(E)
     struct All_variables *E;

{
    void v_from_vector();
    void v_from_vector_pseudo_surf();
    void p_to_nodes();

    int cycles;

    cycles=E->control.p_iterations;

    /* Solve for velocity and pressure, correct for bc's */

    solve_Ahat_p_fhat(E,E->U,E->P,E->F,E->control.accuracy,&cycles);

    if(E->control.pseudo_free_surf)
        v_from_vector_pseudo_surf(E);
    else
        v_from_vector(E);

    p_to_nodes(E,E->P,E->NP,E->mesh.levmax);
}

/* ========================================================================= */
static double momentum_eqn_residual(struct All_variables *E,
                                    double *V, double *P, double *F)
{
    /* Compute the norm of (F - grad(P) - K*V)
     * This norm is ~= E->monitor.momentum_residual */
    void assemble_del2_u();
    void assemble_grad_p();
    void strip_bcs_from_residual();
    double global_v_norm2();

    int i, m;
    double *r1, *r2;
    double res;
    const int lev = E->mesh.levmax;
    const int neq = E->lmesh.neq;

    r1 = malloc((neq+1)*sizeof(double));
    r2 = malloc((neq+1)*sizeof(double));

    /* r2 = F - grad(P) - K*V */
    assemble_grad_p(E, P, E->u1, lev);
    assemble_del2_u(E, V, r1, lev, 1);
    for(i=0; i<neq; i++)
      r2[i] = F[i] - E->u1[i] - r1[i];

    strip_bcs_from_residual(E, r2, lev);

    res = sqrt(global_v_norm2(E, r2));

    free(r1);
    free(r2);
    return(res);
}


static void print_convergence_progress(struct All_variables *E,
                                       int count, double time0,
                                       double v_norm, double p_norm,
                                       double dv, double dp,
                                       double div)
{
    double CPU_time0(), t;
    t = CPU_time0() - time0;

    fprintf(E->fp, "(%03d) %5.1f s v=%e p=%e "
            "div/v=%.2e dv/v=%.2e dp/p=%.2e step %d\n",
            count, t, v_norm, p_norm, div, dv, dp,
            E->monitor.solution_cycles);
    fprintf(stderr, "(%03d) %5.1f s v=%e p=%e "
            "div/v=%.2e dv/v=%.2e dp/p=%.2e step %d\n",
            count, t, v_norm, p_norm, div, dv, dp,
            E->monitor.solution_cycles);
    fflush(stderr);
}


static int keep_iterating(struct All_variables *E,
                          double acc, int converging)
{
    const int required_converging_loops = 2;

    if(E->control.check_continuity_convergence)
        return (E->monitor.incompressibility > acc) ||
	    (converging < required_converging_loops);
    else
        return (E->monitor.incompressibility > acc) &&
	    (converging < required_converging_loops);
}


static void solve_Ahat_p_fhat(struct All_variables *E,
                               double *V, double *P, double *F,
                               double imp, int *steps_max)
{
    if(E->control.inv_gruneisen == 0)
        solve_Ahat_p_fhat_CG(E, V, P, F, imp, steps_max);
    else {
        if(strcmp(E->control.uzawa, "cg") == 0)
            solve_Ahat_p_fhat_iterCG(E, V, P, F, imp, steps_max);
        else if(strcmp(E->control.uzawa, "bicg") == 0)
            solve_Ahat_p_fhat_BiCG(E, V, P, F, imp, steps_max);
        else
            myerror(E, "Error: unknown Uzawa iteration\n");
    }
}


/* Solve incompressible Stokes flow using
 * conjugate gradient (CG) iterations
 */

static void solve_Ahat_p_fhat_CG(struct All_variables *E,
                                 double *V, double *P, double *FF,
                                 double imp, int *steps_max)
{
    int m, j, count, valid, lev, npno, neq;

    double *r1, *r2, *z1, *s1, *s2, *cu;
    double *F;
    double *shuffle;
    double alpha, delta, r0dotz0, r1dotz1;
    double v_res;
    double inner_imp;
    double global_pdot();
    double global_v_norm2(), global_p_norm2(), global_div_norm2();

    double time0, CPU_time0();
    double v_norm, p_norm;
    double dvelocity, dpressure;
    int converging;
    void assemble_c_u();
    void assemble_div_u();
    void assemble_del2_u();
    void assemble_grad_p();
    void strip_bcs_from_residual();
    int  solve_del2_u();
    void parallel_process_termination();
    void v_from_vector();
    void v_from_vector_pseudo_surf();
    void assign_v_to_vector();

    inner_imp = imp * E->control.inner_accuracy_scale; /* allow for different innner loop accuracy */

    npno = E->lmesh.npno;
    neq = E->lmesh.neq;
    lev = E->mesh.levmax;

    F = (double *)malloc(neq*sizeof(double));
    r1 = (double *)malloc((npno+1)*sizeof(double));
    r2 = (double *)malloc((npno+1)*sizeof(double));
    z1 = (double *)malloc((npno+1)*sizeof(double));
    s1 = (double *)malloc((npno+1)*sizeof(double));
    s2 = (double *)malloc((npno+1)*sizeof(double));
    cu = (double *)malloc((npno+1)*sizeof(double));

    time0 = CPU_time0();
    count = 0;
    v_res = E->monitor.fdotf;

    /* copy the original force vector since we need to keep it intact
       between iterations */
    for(j=0;j<neq;j++)
      F[j] = FF[j];


    /* calculate the contribution of compressibility in the continuity eqn */
    if(E->control.inv_gruneisen != 0) {
      for(j=1;j<=npno;j++)
        cu[j] = 0.0;

      assemble_c_u(E, V, cu, lev);
    }


    /* calculate the initial velocity residual */
    /* In the compressible case, the initial guess of P might be bad.
     * Do not correct V with it. */
    if(E->control.inv_gruneisen == 0)
        initial_vel_residual(E, V, P, F, inner_imp*v_res);


    /* initial residual r1 = div(V) */
    assemble_div_u(E, V, r1, lev);

    /* add the contribution of compressibility to the initial residual */
    if(E->control.inv_gruneisen != 0)
      for(j=1;j<=npno;j++)
          r1[j] += cu[j];

    E->monitor.vdotv = global_v_norm2(E, V);
    E->monitor.incompressibility = sqrt(global_div_norm2(E, r1)
                                        / (1e-32 + E->monitor.vdotv));

    v_norm = sqrt(E->monitor.vdotv);
    p_norm = sqrt(E->monitor.pdotp);
    dvelocity = 1.0;
    dpressure = 1.0;
    converging = 0;

    if (E->control.print_convergence && E->parallel.me==0)  {
        print_convergence_progress(E, count, time0,
                                   v_norm, p_norm,
                                   dvelocity, dpressure,
                                   E->monitor.incompressibility);
    }

  
    r0dotz0 = 0;

    while( (count < *steps_max) && keep_iterating(E, imp, converging) ) {
        /* require two consecutive converging iterations to quit the while-loop */

        /* preconditioner BPI ~= inv(K), z1 = BPI*r1 */
        for(j=1; j<=npno; j++)
          z1[j] = E->BPI[lev][j] * r1[j];


        /* r1dotz1 = <r1, z1> */
        r1dotz1 = global_pdot(E, r1, z1, lev);
        assert(r1dotz1 != 0.0  /* Division by zero in head of incompressibility iteration */);

        /* update search direction */
        if(count == 0)
            for(j=1; j<=npno; j++)
                s2[j] = z1[j];
        else {
            /* s2 = z1 + s1 * <r1,z1>/<r0,z0> */
            delta = r1dotz1 / r0dotz0;
            for(j=1; j<=npno; j++)
              s2[j] = z1[j] + delta * s1[j];
        }

        /* solve K*u1 = grad(s2) for u1 */
        assemble_grad_p(E, s2, F, lev);
        valid = solve_del2_u(E, E->u1, F, inner_imp*v_res, lev);
        if(!valid && (E->parallel.me==0)) {
            fputs("Warning: solver not converging! 1\n", stderr);
            fputs("Warning: solver not converging! 1\n", E->fp);
        }
        strip_bcs_from_residual(E, E->u1, lev);


        /* F = div(u1) */
        assemble_div_u(E, E->u1, F, lev);


        /* alpha = <r1, z1> / <s2, F> */
        alpha = r1dotz1 / global_pdot(E, s2, F, lev);


        /* r2 = r1 - alpha * div(u1) */
        for(j=1; j<=npno; j++)
          r2[j] = r1[j] - alpha * F[j];

        /* P = P + alpha * s2 */
        for(j=1; j<=npno; j++)
          P[j] += alpha * s2[j];

        /* V = V - alpha * u1 */
        for(j=0; j<neq; j++)
          V[j] -= alpha * E->u1[j];


        /* compute velocity and incompressibility residual */
        E->monitor.vdotv = global_v_norm2(E, V);
        E->monitor.pdotp = global_p_norm2(E, P);
        v_norm = sqrt(E->monitor.vdotv);
        p_norm = sqrt(E->monitor.pdotp);
        dvelocity = 
          alpha * sqrt(global_v_norm2(E, E->u1) / (1e-32 + E->monitor.vdotv));
        dpressure = 
          alpha * sqrt(global_p_norm2(E, s2) / (1e-32 + E->monitor.pdotp));

       

        assemble_div_u(E, V, z1, lev);
        if(E->control.inv_gruneisen != 0)
          for(j=1;j<=npno;j++)
            z1[j] += cu[j];
        E->monitor.incompressibility = sqrt(global_div_norm2(E, z1)
                                            / (1e-32 + E->monitor.vdotv));

        count++;


        if (E->control.print_convergence && E->parallel.me==0)  {
            print_convergence_progress(E, count, time0,
                                       v_norm, p_norm,
                                       dvelocity, dpressure,
                                       E->monitor.incompressibility);
        }

        if(!valid){
          /* reset consecutive converging iterations */
          converging = 0;
        }else{
          /* how many consecutive converging iterations? */
          if(E->control.check_pressure_convergence) {
              /* check dv and dp */
              if(dvelocity < imp && dpressure < imp)
                  converging++;
              else
                  converging = 0;
          }else{
              /* check dv only */
              if(dvelocity < imp)
                  converging++;
              else
                  converging = 0;
          }
          
        }

        /* shift array pointers */
        shuffle = s1;
        s1 = s2;
        s2 = shuffle;

        shuffle = r1;
        r1 = r2;
        r2 = shuffle;

        /* shift <r0, z0> = <r1, z1> */
        r0dotz0 = r1dotz1;
	if((E->sphere.caps == 12) && (E->control.inner_remove_rigid_rotation)){
	  /* allow for removal of net rotation at each iterative step
	     (expensive) */
	  if(E->control.pseudo_free_surf) /* move from U to V */
	    v_from_vector_pseudo_surf(E);
	  else
	    v_from_vector(E);
	  remove_rigid_rot(E);	/* correct V */
	  assign_v_to_vector(E); /* assign V to U */
	}
    } /* end loop for conjugate gradient */

    assemble_div_u(E, V, z1, lev);
    if(E->control.inv_gruneisen != 0)
      for(j=1;j<=npno;j++)
        z1[j] += cu[j];


    free((void *) F);
    free((void *) r1);
    free((void *) r2);
    free((void *) z1);
    free((void *) s1);
    free((void *) s2);
    free((void *) cu);

    *steps_max=count;
}

/* Solve compressible Stokes flow using
 * bi-conjugate gradient stablized (BiCG-stab) iterations
 */
static void solve_Ahat_p_fhat_BiCG(struct All_variables *E,
                                   double *V, double *P, double *FF,
                                   double imp, int *steps_max)
{
    void assemble_div_rho_u();
    void assemble_del2_u();
    void assemble_grad_p();
    void strip_bcs_from_residual();
    int  solve_del2_u();
    void parallel_process_termination();

    double global_pdot();
    double global_v_norm2(), global_p_norm2(), global_div_norm2();
    double CPU_time0();

    int npno, neq;
    int m, j, count, lev;
    int valid;

    double alpha, beta, omega,inner_imp;
    double r0dotrt, r1dotrt;
    double v_norm, p_norm;
    double dvelocity, dpressure;
    int converging;

    double *F;
    double *r1, *r2, *pt, *p1, *p2;
    double *rt, *v0, *s0, *st, *t0;
    double *u0;
    double *shuffle;

    double time0, v_res;
    
    inner_imp = imp * E->control.inner_accuracy_scale; /* allow for different innner loop accuracy */

    npno = E->lmesh.npno;
    neq = E->lmesh.neq;
    lev = E->mesh.levmax;

    F = (double *)malloc(neq*sizeof(double));
    r1 = (double *)malloc((npno+1)*sizeof(double));
    r2 = (double *)malloc((npno+1)*sizeof(double));
    pt = (double *)malloc((npno+1)*sizeof(double));
    p1 = (double *)malloc((npno+1)*sizeof(double));
    p2 = (double *)malloc((npno+1)*sizeof(double));
    rt = (double *)malloc((npno+1)*sizeof(double));
    v0 = (double *)malloc((npno+1)*sizeof(double));
    s0 = (double *)malloc((npno+1)*sizeof(double));
    st = (double *)malloc((npno+1)*sizeof(double));
    t0 = (double *)malloc((npno+1)*sizeof(double));
    u0 = (double *)malloc(neq*sizeof(double));

    time0 = CPU_time0();
    count = 0;
    v_res = E->monitor.fdotf;

    /* copy the original force vector since we need to keep it intact
       between iterations */
    for(j=0;j<neq;j++)
        F[j] = FF[j];

    /* calculate the initial velocity residual */
    initial_vel_residual(E, V, P, F, inner_imp*v_res);


    /* initial residual r1 = div(rho_ref*V) */
    assemble_div_rho_u(E, V, r1, lev);

    E->monitor.vdotv = global_v_norm2(E, V);
    E->monitor.incompressibility = sqrt(global_div_norm2(E, r1)
                                        / (1e-32 + E->monitor.vdotv));

    v_norm = sqrt(E->monitor.vdotv);
    p_norm = sqrt(E->monitor.pdotp);
    dvelocity = 1.0;
    dpressure = 1.0;
    converging = 0;


    if (E->control.print_convergence && E->parallel.me==0)  {
        print_convergence_progress(E, count, time0,
                                   v_norm, p_norm,
                                   dvelocity, dpressure,
                                   E->monitor.incompressibility);
    }

    /* initial conjugate residual rt = r1 */
    for(j=1; j<=npno; j++)
        rt[j] = r1[j];


    valid = 1;
    r0dotrt = alpha = omega = 0;

    while( (count < *steps_max) && keep_iterating(E, imp, converging) ) {
        /* require two consecutive converging iterations to quit the while-loop */

        /* r1dotrt = <r1, rt> */
        r1dotrt = global_pdot(E, r1, rt, lev);
        if(r1dotrt == 0.0) {
            /* XXX: can we resume the computation when BiCGstab failed? */
            fprintf(E->fp, "BiCGstab method failed!!\n");
            fprintf(stderr, "BiCGstab method failed!!\n");
            parallel_process_termination();
        }

        /* update search direction */
        if(count == 0)
            for(j=1; j<=npno; j++)
                p2[j] = r1[j];
        else {
            /* p2 = r1 + <r1,rt>/<r0,rt> * alpha/omega * (p1 - omega*v0) */
            beta = (r1dotrt / r0dotrt) * (alpha / omega);
            for(j=1; j<=npno; j++)
              p2[j] = r1[j] + beta*(p1[j] - omega*v0[j]);
        }


        /* preconditioner BPI ~= inv(K), pt = BPI*p2 */
        for(j=1; j<=npno; j++)
            pt[j] = E->BPI[lev][j] * p2[j];


        /* solve K*u0 = grad(pt) for u1 */
        assemble_grad_p(E, pt, F, lev);
        valid = solve_del2_u(E, u0, F, inner_imp*v_res, lev);
        if(!valid && (E->parallel.me==0)) {
            fputs("Warning: solver not converging! 1\n", stderr);
            fputs("Warning: solver not converging! 1\n", E->fp);
        }
        strip_bcs_from_residual(E, u0, lev);

        /* v0 = div(rho_ref*u0) */
        assemble_div_rho_u(E, u0, v0, lev);

        /* alpha = r1dotrt / <rt, v0> */
        alpha = r1dotrt / global_pdot(E, rt, v0, lev);

        /* s0 = r1 - alpha * v0 */
        for(j=1; j<=npno; j++)
            s0[j] = r1[j] - alpha * v0[j];

        /* preconditioner BPI ~= inv(K), st = BPI*s0 */
        for(j=1; j<=npno; j++)
            st[j] = E->BPI[lev][j] * s0[j];

        /* solve K*u1 = grad(st) for u1 */
        assemble_grad_p(E, st, F, lev);
        valid = solve_del2_u(E, E->u1, F, inner_imp*v_res, lev);
        if(!valid && (E->parallel.me==0)) {
            fputs("Warning: solver not converging! 2\n", stderr);
            fputs("Warning: solver not converging! 2\n", E->fp);
        }
        strip_bcs_from_residual(E, E->u1, lev);


        /* t0 = div(rho_ref * u1) */
        assemble_div_rho_u(E, E->u1, t0, lev);

        /* omega = <t0, s0> / <t0, t0> */
        omega = global_pdot(E, t0, s0, lev) / global_pdot(E, t0, t0, lev);

        /* r2 = s0 - omega * t0 */
        for(j=1; j<=npno; j++)
            r2[j] = s0[j] - omega * t0[j];

        /* P = P + alpha * pt + omega * st */
        for(j=1; j<=npno; j++)
            s0[j] = alpha * pt[j] + omega * st[j];

        for(j=1; j<=npno; j++)
            P[j] += s0[j];

        /* V = V - alpha * u0 - omega * u1 */
        for(j=0; j<neq; j++)
            F[j] = alpha * u0[j] + omega * E->u1[j];

        for(j=0; j<neq; j++)
            V[j] -= F[j];


        /* compute velocity and incompressibility residual */
        E->monitor.vdotv = global_v_norm2(E, V);
        E->monitor.pdotp = global_p_norm2(E, P);
        v_norm = sqrt(E->monitor.vdotv);
        p_norm = sqrt(E->monitor.pdotp);
        dvelocity = sqrt(global_v_norm2(E, F) / (1e-32 + E->monitor.vdotv));
        dpressure = sqrt(global_p_norm2(E, s0) / (1e-32 + E->monitor.pdotp));


	

        assemble_div_rho_u(E, V, t0, lev);
        E->monitor.incompressibility = sqrt(global_div_norm2(E, t0)
                                            / (1e-32 + E->monitor.vdotv));




        count++;

        if(E->control.print_convergence && E->parallel.me==0) {
            print_convergence_progress(E, count, time0,
                                       v_norm, p_norm,
                                       dvelocity, dpressure,
                                       E->monitor.incompressibility);
        }

      if(!valid){
        /* reset consecutive converging iterations */
        converging = 0;
      }else{
        /* how many consecutive converging iterations? */
        if(E->control.check_pressure_convergence) {
            /* check dv and dp */
            if(dvelocity < imp && dpressure < imp)
                converging++;
            else
                converging = 0;
        }else{
            /* check dv only */
            if(dvelocity < imp)
                converging++;
            else
                converging = 0;
        }
        
      }

      /* shift array pointers */
      shuffle = p1;
      p1 = p2;
      p2 = shuffle;

      shuffle = r1;
      r1 = r2;
      r2 = shuffle;

      /* shift <r0, rt> = <r1, rt> */
      r0dotrt = r1dotrt;

    } /* end loop for conjugate gradient */


    free((void *) F);
    free((void *) r1);
    free((void *) r2);
    free((void *) pt);
    free((void *) p1);
    free((void *) p2);
    free((void *) rt);
    free((void *) v0);
    free((void *) s0);
    free((void *) st);
    free((void *) t0);
    free((void *) u0);

    *steps_max=count;
}

/* Solve compressible Stokes flow using
 * conjugate gradient (CG) iterations with an outer iteration
 */
static void solve_Ahat_p_fhat_iterCG(struct All_variables *E,
                                     double *V, double *P, double *F,
                                     double imp, int *steps_max)
{
    int m, i;
    int cycles, num_of_loop;
    double relative_err_v, relative_err_p;
    double *old_v, *old_p,*diff_v,*diff_p;
    double div_res;
    const int npno = E->lmesh.npno;
    const int neq = E->lmesh.neq;
    const int lev = E->mesh.levmax;

    double global_v_norm2(),global_p_norm2();
    double global_div_norm2();
    void assemble_div_rho_u();
    
    old_v = (double *)malloc(neq*sizeof(double));
    diff_v = (double *)malloc(neq*sizeof(double));
    old_p = (double *)malloc((npno+1)*sizeof(double));
    diff_p = (double *)malloc((npno+1)*sizeof(double));

    cycles = E->control.p_iterations;

    initial_vel_residual(E, V, P, F,
                         imp * E->control.inner_accuracy_scale * E->monitor.fdotf);

    div_res = 1.0;
    relative_err_v = 1.0;
    relative_err_p = 1.0;
    num_of_loop = 0;

    while((relative_err_v >= imp || relative_err_p >= imp) &&
          (div_res > imp) &&
          (num_of_loop <= E->control.compress_iter_maxstep)) {

          for(i=0;i<neq;i++) 
            old_v[i] = V[i];
          for(i=1;i<=npno;i++) 
            old_p[i] = P[i];

        solve_Ahat_p_fhat_CG(E, V, P, F, imp, &cycles);

        /* compute norm of div(rho*V) */
        assemble_div_rho_u(E, V, E->u1, lev);
        div_res = sqrt(global_div_norm2(E, E->u1) / (1e-32 + E->monitor.vdotv));

        for(i=0;i<neq;i++) 
          diff_v[i] = V[i] - old_v[i];

        relative_err_v = sqrt( global_v_norm2(E,diff_v) /
                               (1.0e-32 + E->monitor.vdotv) );

        for(i=1;i<=npno;i++) 
          diff_p[i] = P[i] - old_p[i];

        relative_err_p = sqrt( global_p_norm2(E,diff_p) /
                               (1.0e-32 + E->monitor.pdotp) );

        if(E->parallel.me == 0) {
            fprintf(stderr, "itercg -- div(rho*v)/v=%.2e dv/v=%.2e and dp/p=%.2e loop %d\n\n", div_res, relative_err_v, relative_err_p, num_of_loop);
            fprintf(E->fp, "itercg -- div(rho*v)/v=%.2e dv/v=%.2e and dp/p=%.2e loop %d\n\n", div_res, relative_err_v, relative_err_p, num_of_loop);
        }

        num_of_loop++;

    } /* end of while */

    free((void *) old_v);
    free((void *) old_p);
    free((void *) diff_v);
    free((void *) diff_p);
}

static void initial_vel_residual(struct All_variables *E,
                                 double *V, double *P, double *F,
                                 double acc)
{
    void assemble_del2_u();
    void assemble_grad_p();
    void strip_bcs_from_residual();
    int  solve_del2_u();

    int neq = E->lmesh.neq;
    int lev = E->mesh.levmax;
    int i, m, valid;

    /* F = F - grad(P) - K*V */
    assemble_grad_p(E, P, E->u1, lev);
    for(i=0; i<neq; i++)
      F[i] = F[i] - E->u1[i];

    assemble_del2_u(E, V, E->u1, lev, 1);
    for(i=0; i<neq; i++)
      F[i] = F[i] - E->u1[i];

    strip_bcs_from_residual(E, F, lev);


    /* solve K*u1 = F for u1 */
    valid = solve_del2_u(E, E->u1, F, acc, lev);
    if(!valid && (E->parallel.me==0)) {
        fputs("Warning: solver not converging! 0\n", stderr);
        fputs("Warning: solver not converging! 0\n", E->fp);
    }
    strip_bcs_from_residual(E, E->u1, lev);


    /* V = V + u1 */
    for(i=0; i<neq; i++)
        V[i] += E->u1[i];
}
