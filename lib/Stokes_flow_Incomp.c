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
#include "petsc_citcoms.h"
#include <stdlib.h>

void myerror(struct All_variables *,char *);

static void solve_Ahat_p_fhat(struct All_variables *E,
                              double **V, double **P, double **F,
                              double imp, int *steps_max);
static void solve_Ahat_p_fhat_CG(struct All_variables *E,
                                 double **V, double **P, double **F,
                                 double imp, int *steps_max);
static void solve_Ahat_p_fhat_BiCG(struct All_variables *E,
                                    double **V, double **P, double **F,
                                    double imp, int *steps_max);
static void solve_Ahat_p_fhat_iterCG(struct All_variables *E,
                                      double **V, double **P, double **F,
                                      double imp, int *steps_max);

static PetscErrorCode solve_Ahat_p_fhat_petsc(struct All_variables *E,
    Vec V, Vec P, Vec F, double imp, int *steps_max);

static PetscErrorCode solve_Ahat_p_fhat_PETSc_Schur(struct All_variables *E,
    double **V, double **P, double **F, double imp, int *steps_max);

static PetscErrorCode solve_Ahat_p_fhat_CG_PETSc(struct All_variables *E, 
    double **V, double **P, double **F, double imp, int *steps_max);

static PetscErrorCode solve_Ahat_p_fhat_BiCG_PETSc(struct All_variables *E,
    double **V, double **P, double **F, double imp, int *steps_max);

static void initial_vel_residual(struct All_variables *E,
                                 double **V, double **P, double **F,
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

    return;
}
/* ========================================================================= */

static double momentum_eqn_residual(struct All_variables *E,
                                    double **V, double **P, double **F)
{
    /* Compute the norm of (F - grad(P) - K*V)
     * This norm is ~= E->monitor.momentum_residual */
    void assemble_del2_u();
    void assemble_grad_p();
    void strip_bcs_from_residual();
    double global_v_norm2();

    int i, m;
    double *r1[NCS], *r2[NCS];
    double res;
    const int lev = E->mesh.levmax;
    const int neq = E->lmesh.neq;

    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        r1[m] = malloc((neq+1)*sizeof(double));
        r2[m] = malloc((neq+1)*sizeof(double));
    }

    /* r2 = F - grad(P) - K*V */
    assemble_grad_p(E, P, E->u1, lev);
    assemble_del2_u(E, V, r1, lev, 1);
    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(i=0; i<neq; i++)
            r2[m][i] = F[m][i] - E->u1[m][i] - r1[m][i];

    strip_bcs_from_residual(E, r2, lev);

    res = sqrt(global_v_norm2(E, r2));

    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        free(r1[m]);
        free(r2[m]);
    }
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
    return;
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

static PetscErrorCode solve_Ahat_p_fhat_petsc(struct All_variables *E,
    Vec V, Vec P, Vec F, double imp, int *steps_max )
{
  PetscFunctionReturn(0);
}

static void solve_Ahat_p_fhat(struct All_variables *E,
                               double **V, double **P, double **F,
                               double imp, int *steps_max)
{
  if(E->control.use_petsc)
  {
    if(E->control.petsc_schur) // use Schur complement reduction
    {
      solve_Ahat_p_fhat_PETSc_Schur(E, V, P, F, imp, steps_max);
    }
    else                       // use the Uzawa algorithm
    {
      if(E->control.inv_gruneisen == 0)
        solve_Ahat_p_fhat_CG_PETSc(E, V, P, F, imp, steps_max);
      else
      {
        if(strcmp(E->control.uzawa, "cg") == 0)
          solve_Ahat_p_fhat_iterCG(E, V, P, F, imp, steps_max);
        else if(strcmp(E->control.uzawa, "bicg") == 0)
          solve_Ahat_p_fhat_BiCG_PETSc(E, V, P, F, imp, steps_max);
        else
          myerror(E, "Error: unknown Uzawa iteration\n");
      }
    }
  }
  else                         // the original non-PETSc CitcomS code
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
}

static PetscErrorCode solve_Ahat_p_fhat_PETSc_Schur(struct All_variables *E,
  double **V, double **P, double **F, double imp, int *steps_max)
{
  PetscErrorCode ierr;

  Mat S;
  ierr = MatCreateSchurComplement(E->K,E->K,E->G,E->D,PETSC_NULL, &S); 
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* Solve incompressible Stokes flow using
 * conjugate gradient (CG) iterations
 */

static void solve_Ahat_p_fhat_CG(struct All_variables *E,
                                 double **V, double **P, double **FF,
                                 double imp, int *steps_max)
{
    int m, j, count, valid, lev, npno, neq;

    double *r1[NCS], *r2[NCS], *z1[NCS], *s1[NCS], *s2[NCS], *cu[NCS];
    double *F[NCS];
    double *shuffle[NCS];
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

    for (m=1; m<=E->sphere.caps_per_proc; m++)   {
        F[m] = (double *)malloc(neq*sizeof(double));
        r1[m] = (double *)malloc((npno+1)*sizeof(double));
        r2[m] = (double *)malloc((npno+1)*sizeof(double));
        z1[m] = (double *)malloc((npno+1)*sizeof(double));
        s1[m] = (double *)malloc((npno+1)*sizeof(double));
        s2[m] = (double *)malloc((npno+1)*sizeof(double));
        cu[m] = (double *)malloc((npno+1)*sizeof(double));
    }

    time0 = CPU_time0();
    count = 0;
    v_res = E->monitor.fdotf;

    /* copy the original force vector since we need to keep it intact
       between iterations */
    for(m=1;m<=E->sphere.caps_per_proc;m++)
        for(j=0;j<neq;j++)
            F[m][j] = FF[m][j];


    /* calculate the contribution of compressibility in the continuity eqn */
    if(E->control.inv_gruneisen != 0) {
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(j=1;j<=npno;j++)
                cu[m][j] = 0.0;

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
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(j=1;j<=npno;j++) {
                r1[m][j] += cu[m][j];
            }

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
        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=1; j<=npno; j++)
                z1[m][j] = E->BPI[lev][m][j] * r1[m][j];


        /* r1dotz1 = <r1, z1> */
        r1dotz1 = global_pdot(E, r1, z1, lev);
        assert(r1dotz1 != 0.0  /* Division by zero in head of incompressibility iteration */);

        /* update search direction */
        if(count == 0)
            for (m=1; m<=E->sphere.caps_per_proc; m++)
                for(j=1; j<=npno; j++)
                    s2[m][j] = z1[m][j];
        else {
            /* s2 = z1 + s1 * <r1,z1>/<r0,z0> */
            delta = r1dotz1 / r0dotz0;
            for(m=1; m<=E->sphere.caps_per_proc; m++)
                for(j=1; j<=npno; j++)
                    s2[m][j] = z1[m][j] + delta * s1[m][j];
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
        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=1; j<=npno; j++)
                r2[m][j] = r1[m][j] - alpha * F[m][j];


        /* P = P + alpha * s2 */
        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=1; j<=npno; j++)
                P[m][j] += alpha * s2[m][j];


        /* V = V - alpha * u1 */
        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=0; j<neq; j++)
                V[m][j] -= alpha * E->u1[m][j];


        /* compute velocity and incompressibility residual */
        E->monitor.vdotv = global_v_norm2(E, V);
        E->monitor.pdotp = global_p_norm2(E, P);
        v_norm = sqrt(E->monitor.vdotv);
        p_norm = sqrt(E->monitor.pdotp);
        dvelocity = alpha * sqrt(global_v_norm2(E, E->u1) / (1e-32 + E->monitor.vdotv));
        dpressure = alpha * sqrt(global_p_norm2(E, s2) / (1e-32 + E->monitor.pdotp));

       

        assemble_div_u(E, V, z1, lev);
        if(E->control.inv_gruneisen != 0)
            for(m=1;m<=E->sphere.caps_per_proc;m++)
                for(j=1;j<=npno;j++) {
                    z1[m][j] += cu[m][j];
            }
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
        for(m=1; m<=E->sphere.caps_per_proc; m++) {
            shuffle[m] = s1[m];
            s1[m] = s2[m];
            s2[m] = shuffle[m];

            shuffle[m] = r1[m];
            r1[m] = r2[m];
            r2[m] = shuffle[m];
        }

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
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(j=1;j<=npno;j++) {
                z1[m][j] += cu[m][j];
            }


    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        free((void *) F[m]);
        free((void *) r1[m]);
        free((void *) r2[m]);
        free((void *) z1[m]);
        free((void *) s1[m]);
        free((void *) s2[m]);
        free((void *) cu[m]);
    }

    *steps_max=count;

    return;
}

/*
 * Implementation of the Conjugate Gradient Uzawa algorithm using PETSc
 * Vec, Mat and KSPSolve
 */
static PetscErrorCode solve_Ahat_p_fhat_CG_PETSc( struct All_variables *E,
				     double **V, double **P, double **F,
				     double imp, int *steps_max )
{
  PetscErrorCode ierr;
  Vec V_k, P_k, s_1, s_2, r_1, r_2, z_1, BPI, FF, Gsk, u_k, Duk, cu;
  PetscReal alpha, delta, r_1_norm, r1dotz1, r0dotz0, s_2_dot_F;
  int i,j,count,m;
  double time0, CPU_time0();
  double v_norm, p_norm, dvelocity, dpressure;
  double inner_imp, v_res;

  int lev = E->mesh.levmax;
  int npno = E->lmesh.npno;
  int neq = E->lmesh.neq;
  int nel = E->lmesh.nel;

  inner_imp = imp * E->control.inner_accuracy_scale; 
  v_res = E->monitor.fdotf;

  time0 = CPU_time0();
  count = 0;

  // Create the force Vec
  ierr = VecCreateMPI( PETSC_COMM_WORLD, neq+1, PETSC_DECIDE, &FF ); 
  CHKERRQ(ierr);
  double *F_tmp;
  ierr = VecGetArray( FF, &F_tmp ); CHKERRQ( ierr );
  for( m = 1; m <= E->sphere.caps_per_proc; m++ ) {
    for( i = 0; i < neq; i++ )
      F_tmp[i] = F[m][i];
  }
  ierr = VecRestoreArray( FF, &F_tmp ); CHKERRQ( ierr );

  // create the pressure vector and initialize it to zero
  ierr = VecCreateMPI( PETSC_COMM_WORLD, nel, PETSC_DECIDE, &P_k ); 
  CHKERRQ(ierr);
  ierr = VecSet( P_k, 0.0 ); CHKERRQ( ierr );

  // create the velocity vector
  ierr = VecCreateMPI( PETSC_COMM_WORLD, neq+1, PETSC_DECIDE, &V_k ); 
  CHKERRQ(ierr);

  // Copy the contents of V into V_k
  PetscScalar *V_k_tmp;
  ierr = VecGetArray( V_k, &V_k_tmp ); CHKERRQ( ierr );
  for( m = 1; m <= E->sphere.caps_per_proc; m++ ) {
    for( i = 0; i <= neq; i++ )
      V_k_tmp[i] = V[m][i];
  }
  ierr = VecRestoreArray( V_k, &V_k_tmp ); CHKERRQ( ierr );

  // PETSc bookkeeping --- create various temporary Vec objects with
  // the appropriate sizes, including the PETSc vec version of E->BPI
  // preconditioner
  ierr = VecCreateMPI( PETSC_COMM_WORLD, npno, PETSC_DECIDE, &r_1 ); 
  CHKERRQ(ierr);
  ierr = VecDuplicate( V_k, &Gsk ); CHKERRQ( ierr );
  ierr = VecDuplicate( V_k, &u_k ); CHKERRQ( ierr );
  ierr = VecDuplicate( r_1, &s_1 ); CHKERRQ( ierr );
  ierr = VecDuplicate( r_1, &s_2 ); CHKERRQ( ierr );
  ierr = VecDuplicate( r_1, &r_2 ); CHKERRQ( ierr );
  ierr = VecDuplicate( r_1, &z_1 ); CHKERRQ( ierr );
  ierr = VecDuplicate( r_1, &cu ); CHKERRQ( ierr );
  ierr = VecDuplicate( r_1, &Duk ); CHKERRQ( ierr );
  ierr = VecDuplicate( r_1, &BPI ); CHKERRQ( ierr );
  PetscReal *bpi;
  ierr = VecGetArray( BPI, &bpi ); CHKERRQ( ierr );
  for( m = 1; m <= E->sphere.caps_per_proc; m++ )
    for( j = 0; j < npno; j++ )
      bpi[j] = E->BPI[lev][m][j+1];
  ierr = VecRestoreArray( BPI, &bpi ); CHKERRQ( ierr );

  /* calculate the contribution of compressibility in the continuity eqn */
  if( E->control.inv_gruneisen != 0 ) {
    ierr = VecSet( cu, 0.0 ); CHKERRQ( ierr );
    assemble_c_u_PETSc( E, V_k, cu, lev );
  }

  /* calculate the initial velocity residual */
  /* In the compressible case, the initial guess of P might be bad.
   * Do not correct V with it. */
  if( E->control.inv_gruneisen == 0 ) {
    initial_vel_residual_PETSc(E, V_k, P_k, FF, inner_imp*v_res);
  }

  /* initial residual r1 = div(V) */
  ierr = MatMult( E->D, V_k, r_1 ); CHKERRQ( ierr );

  /* add the contribution of compressibility to the initial residual */
  if( E->control.inv_gruneisen != 0 ) {
    // r_1 += cu
    ierr = VecAXPY( r_1, 1.0, cu ); CHKERRQ( ierr );
  }

  E->monitor.vdotv = global_v_norm2_PETSc( E, V_k );
  E->monitor.incompressibility = sqrt( global_div_norm2_PETSc( E, r_1 )
                                       / (1e-32 + E->monitor.vdotv) );

  v_norm = sqrt( E->monitor.vdotv );
  p_norm = sqrt( E->monitor.pdotp );
  dvelocity = 1.0;
  dpressure = 1.0;

  if( E->control.print_convergence && E->parallel.me == 0 ) {
    print_convergence_progress( E, count, time0, 
                                v_norm, p_norm,
                                dvelocity, dpressure,
                                E->monitor.incompressibility );
  }

  
  r0dotz0 = 0;
  ierr = VecNorm( r_1, NORM_2, &r_1_norm ); CHKERRQ( ierr );

  while( (r_1_norm > E->control.petsc_uzawa_tol) && (count < *steps_max) )
    {
      
      /* preconditioner BPI ~= inv(K), z1 = BPI*r1 */
      ierr = VecPointwiseMult( z_1, BPI, r_1 ); CHKERRQ( ierr );

      /* r1dotz1 = <r1, z1> */
      ierr = VecDot( r_1, z_1, &r1dotz1 ); CHKERRQ( ierr );
      assert( r1dotz1 != 0.0  /* Division by zero in head of incompressibility
                                 iteration */);

      /* update search direction */
      if( count == 0 )
	    {
	      // s_2 = z_1
	      ierr = VecCopy( z_1, s_2 ); CHKERRQ( ierr ); // s2 = z1
	    }
      else
	    {
	      // s2 = z1 + s1 * <r1,z1>/<r0,z0>
	      delta = r1dotz1 / r0dotz0;
	      ierr = VecWAXPY( s_2, delta, s_1, z_1 ); CHKERRQ( ierr );
	    }

      // Solve K*u_k = grad(s_2) for u_k
      ierr = MatMult( E->G, s_2, Gsk ); CHKERRQ( ierr );
      ierr = KSPSolve( E->ksp, Gsk, u_k ); CHKERRQ( ierr );
      strip_bcs_from_residual_PETSc( E, u_k, lev );

      // Duk = D*u_k ( D*u_k is the same as div(u_k) )
      ierr = MatMult( E->D, u_k, Duk ); CHKERRQ( ierr );

      // alpha = <r1,z1> / <s2,F>
      ierr = VecDot( s_2, Duk, &s_2_dot_F ); CHKERRQ( ierr );
      alpha = r1dotz1 / s_2_dot_F;

      // r2 = r1 - alpha * div(u_k)
      ierr = VecWAXPY( r_2, -1.0*alpha, Duk, r_1 ); CHKERRQ( ierr );

      // P = P + alpha * s_2
      ierr = VecAXPY( P_k, 1.0*alpha, s_2 ); CHKERRQ( ierr );
      
      // V = V - alpha * u_1
      ierr = VecAXPY( V_k, -1.0*alpha, u_k ); CHKERRQ( ierr );
      //strip_bcs_from_residual_PETSc( E, V_k, E->mesh.levmax );

      /* compute velocity and incompressibility residual */
      E->monitor.vdotv = global_v_norm2_PETSc( E, V_k );
      E->monitor.pdotp = global_p_norm2_PETSc( E, P_k );
      v_norm = sqrt( E->monitor.vdotv );
      p_norm = sqrt( E->monitor.pdotp );
      dvelocity = alpha * sqrt( global_v_norm2_PETSc( E, u_k ) /
                                (1e-32 + E->monitor.vdotv) );
      dpressure = alpha * sqrt( global_p_norm2_PETSc( E, s_2 ) /
                                (1e-32 + E->monitor.pdotp) );

      // compute the updated value of z_1, z1 = div(V) 
      ierr = MatMult( E->D, V_k, z_1 ); CHKERRQ( ierr );
      if( E->control.inv_gruneisen != 0 )
	    {
        // z_1 += cu
        ierr = VecAXPY( z_1, 1.0, cu ); CHKERRQ( ierr );
      }

      E->monitor.incompressibility = sqrt( global_div_norm2_PETSc( E, z_1 )
                                           / (1e-32 + E->monitor.vdotv) );

      count++;

      if( E->control.print_convergence && E->parallel.me == 0 ) {
        print_convergence_progress( E, count, time0,
                                    v_norm, p_norm,
                                    dvelocity, dpressure,
                                    E->monitor.incompressibility );
      }

      /* shift array pointers */
      ierr = VecSwap( s_2, s_1 ); CHKERRQ( ierr );
      ierr = VecSwap( r_2, r_1 ); CHKERRQ( ierr );

      /* shift <r0, z0> = <r1, z1> */
      r0dotz0 = r1dotz1;

      // recompute the norm
      ierr = VecNorm( r_1, NORM_2, &r_1_norm ); CHKERRQ( ierr );

    } /* end loop for conjugate gradient */

  // converged. now copy the converged values of V_k and P_k into V and P
  PetscReal *P_tmp, *V_tmp;
  ierr = VecGetArray( V_k, &V_tmp ); CHKERRQ( ierr );
  for( m = 1; m <= E->sphere.caps_per_proc; ++m ) {
    for( i = 0; i <= neq; i++ )
      V[m][i] = V_tmp[i];
  }
  ierr = VecRestoreArray( V_k, &V_tmp ); CHKERRQ( ierr );
  
  ierr = VecGetArray( P_k, &P_tmp ); CHKERRQ( ierr );
  for( m = 1; m <= E->sphere.caps_per_proc; ++m ) {
    for( i = 0; i < nel; i++ )
      P[m][i+1] = P_tmp[i]; 
  }
  ierr = VecRestoreArray( P_k, &P_tmp ); CHKERRQ( ierr );

  // PETSc cleanup of all temporary Vec objects
  ierr = VecDestroy( &V_k ); CHKERRQ( ierr );
  ierr = VecDestroy( &P_k ); CHKERRQ( ierr );
  ierr = VecDestroy( &s_1 ); CHKERRQ( ierr );
  ierr = VecDestroy( &s_2 ); CHKERRQ( ierr );
  ierr = VecDestroy( &r_1 ); CHKERRQ( ierr );
  ierr = VecDestroy( &r_2 ); CHKERRQ( ierr );
  ierr = VecDestroy( &z_1 ); CHKERRQ( ierr );
  ierr = VecDestroy( &BPI ); CHKERRQ( ierr );
  ierr = VecDestroy( &FF );  CHKERRQ( ierr );
  ierr = VecDestroy( &Gsk ); CHKERRQ( ierr );
  ierr = VecDestroy( &u_k ); CHKERRQ( ierr );
  ierr = VecDestroy( &Duk ); CHKERRQ( ierr );

  *steps_max = count;

  PetscFunctionReturn(0);
}

/*
 * BiCGstab for compressible Stokes flow using PETSc Vec, Mat and KSPSolve
 */
static PetscErrorCode solve_Ahat_p_fhat_BiCG_PETSc( struct All_variables *E,
					  double **V, double **P, double **F,
					  double imp, int *steps_max )
{
  PetscErrorCode ierr;
  Vec FF, P0, p1, p2, pt, r1, r2, rt, s0, st, t0, u0, u1, V0, BPI, v0;

  PetscReal alpha, omega, beta;
  PetscReal r1_norm, r1dotrt, r0dotrt, rtdotV0, t0dots0, t0dott0;

  int i,j,k,m, count;

  double time0, CPU_time0();
  double v_norm, p_norm, inner_imp, v_res, dvelocity, dpressure;

  int lev = E->mesh.levmax;
  int npno = E->lmesh.npno;
  int neq = E->lmesh.neq;
  int nel = E->lmesh.nel;
  
  // Create the force Vec
  ierr = VecCreateMPI( PETSC_COMM_WORLD, neq+1, PETSC_DECIDE, &FF ); 
  CHKERRQ(ierr);
  double *F_tmp;
  ierr = VecGetArray( FF, &F_tmp ); CHKERRQ( ierr );
  for( m=1; m<=E->sphere.caps_per_proc; ++m ) {
    for( i = 0; i < neq; i++ )
      F_tmp[i] = F[m][i];
  }
  ierr = VecRestoreArray( FF, &F_tmp ); CHKERRQ( ierr );

  inner_imp = imp * E->control.inner_accuracy_scale;
  time0 = CPU_time0();
  count = 0;
  v_res = E->monitor.fdotf;


  // create the pressure vector and initialize it to zero
  ierr = VecCreateMPI( PETSC_COMM_WORLD, nel, PETSC_DECIDE, &P0 ); 
  CHKERRQ(ierr);
  ierr = VecSet( P0, 0.0 ); CHKERRQ( ierr );

  // create the velocity vector
  ierr = VecCreateMPI( PETSC_COMM_WORLD, neq+1, PETSC_DECIDE, &V0 ); 
  CHKERRQ(ierr);

  // Copy the contents of V into V0
  PetscScalar *V0_tmp;
  ierr = VecGetArray( V0, &V0_tmp ); CHKERRQ( ierr );
  for( m = 1; m <= E->sphere.caps_per_proc; m++ ) {
    for( i = 0; i <= neq; i++ )
      V0_tmp[i] = V[m][i];
  }
  ierr = VecRestoreArray( V0, &V0_tmp ); CHKERRQ( ierr );


  ierr = VecDuplicate( V0, &u0 ); CHKERRQ( ierr );
  ierr = VecDuplicate( V0, &u1 ); CHKERRQ( ierr );

  /* calculate the initial velocity residual */
  initial_vel_residual_PETSc( E, V0, P0, FF, inner_imp*v_res );

  /* initial residual r1 = div(rho_ref*V) */
  ierr = VecCreateMPI( PETSC_COMM_WORLD, npno, PETSC_DECIDE, &r1 ); 

  CHKERRQ(ierr);
  ierr = MatMult( E->DC, V0, r1 ); CHKERRQ( ierr );

  E->monitor.vdotv = global_v_norm2_PETSc( E, V0 );
  E->monitor.incompressibility = sqrt( global_div_norm2_PETSc( E, r1 )
                                       / (1e-32 + E->monitor.vdotv) );
  v_norm = sqrt( E->monitor.vdotv );
  p_norm = sqrt( E->monitor.pdotp );
  dvelocity = 1.0;
  dpressure = 1.0;

  if( E->control.print_convergence && E->parallel.me == 0 ) {
    print_convergence_progress( E, count, time0,
                                v_norm, p_norm,
                                dvelocity, dpressure,
                                E->monitor.incompressibility );
  }

  // create all the vectors for later use
  ierr = VecDuplicate( r1, &rt ); CHKERRQ( ierr );
  ierr = VecDuplicate( r1, &p1 ); CHKERRQ( ierr );
  ierr = VecDuplicate( r1, &p2 ); CHKERRQ( ierr );
  ierr = VecDuplicate( r1, &BPI ); CHKERRQ( ierr );
  ierr = VecDuplicate( r1, &pt ); CHKERRQ( ierr );
  ierr = VecDuplicate( r1, &s0 ); CHKERRQ( ierr );
  ierr = VecDuplicate( r1, &st ); CHKERRQ( ierr );
  ierr = VecDuplicate( r1, &t0 ); CHKERRQ( ierr );
  ierr = VecDuplicate( r1, &r2 ); CHKERRQ( ierr );
  ierr = VecDuplicate( r1, &v0 ); CHKERRQ( ierr );

  /* initial conjugate residual rt = r1 */
  ierr = VecCopy( r1, rt ); CHKERRQ( ierr );

  /* BPI ~= K inverse */
  PetscReal *bpi;
  ierr = VecGetArray( BPI, &bpi ); CHKERRQ( ierr );
  for( m = 1; m <= E->sphere.caps_per_proc; m++ )
    for( j = 0; j < npno; j++ )
      bpi[j] = E->BPI[lev][m][j+1];
  ierr = VecRestoreArray( BPI, &bpi ); CHKERRQ( ierr );

  r0dotrt = alpha = omega = 0.0;

  ierr = VecNorm( r1, NORM_2, &r1_norm ); CHKERRQ( ierr );

  while( (r1_norm > E->control.petsc_uzawa_tol) && (count < *steps_max) )
  {

    /* r1dotrt = <r1, rt> */
    // r1dotrt = global_pdot( E, r1, rt, lev )
    ierr = VecDot( r1, rt, &r1dotrt ); CHKERRQ( ierr );

    if( r1dotrt == 0.0 ) {
      fprintf( E->fp, "BiCGstab method failed!!\n" );
      fprintf( stderr, "BiCGstab method failed!!\n" );
      parallel_process_termination();
    }

    /* update search direction */
    if( count == 0 ) {
      ierr = VecCopy( r1, p2 ); CHKERRQ( ierr );
    }
    else {
      /* p2 = r1 + <r1,rt>/<r0,rt> * alpha/omega * (p1 - omega*v0) */
      beta = (r1dotrt/r0dotrt)*(alpha/omega);
      ierr = VecAXPY( p1, -1.0*omega, v0); CHKERRQ( ierr );
      ierr = VecWAXPY( p2, beta, p1, r1 ); CHKERRQ( ierr );
    }

    /* preconditioner BPI ~= inv(K), pt = BPI*p2 */
    ierr = VecPointwiseMult( pt, BPI, p2 ); CHKERRQ( ierr );

    /* solve K*u0 = grad(pt) for u1 */
    ierr = MatMult( E->G, pt, FF ); CHKERRQ( ierr );
    ierr = KSPSolve( E->ksp, FF, u0 ); CHKERRQ( ierr );
    strip_bcs_from_residual_PETSc( E, u0, lev );

    /* v0 = div(rho_ref*u0) */
    ierr = MatMult( E->DC, u0, v0 ); CHKERRQ( ierr );
    
    /* alpha = r1dotrt / <rt, v0> */
    ierr = VecDot( rt, v0, &rtdotV0 ); CHKERRQ( ierr );
    alpha = r1dotrt / rtdotV0;

    /* s0 = r1 - alpha * v0 */
    ierr = VecWAXPY( s0, -1.0*alpha, v0, r1 ); CHKERRQ( ierr );

    /* preconditioner BPI ~= inv(K), st = BPI*s0 */
    ierr = VecPointwiseMult( st, BPI, s0 ); CHKERRQ( ierr );

    /* solve K*u1 = grad(st) for u1*/
    ierr = MatMult( E->G, st, FF ); CHKERRQ( ierr );
    ierr = KSPSolve( E->ksp, FF, u1 ); CHKERRQ( ierr );
    strip_bcs_from_residual_PETSc( E, u1, lev );

    /* t0 = div(rho_ref*u1) */
    ierr = MatMult( E->DC, u1, t0 ); CHKERRQ( ierr );

    /* omega = <t0, s0> / <t0, t0> */
    ierr = VecDot( t0, s0, &t0dots0 ); CHKERRQ( ierr );
    ierr = VecDot( t0, t0, &t0dott0 ); CHKERRQ( ierr );
    omega = t0dots0 / t0dott0;

    /* r2 = s0 - omega * t0 */
    ierr = VecWAXPY( r2, -1.0*omega, t0, s0 ); CHKERRQ( ierr );

    /* P = P + alpha * pt + omega * st */
    ierr = VecAXPBY( st, alpha, omega, pt ); CHKERRQ( ierr );
    ierr = VecAXPY( P0, 1.0, st ); CHKERRQ( ierr );

    /* V = V - alpha * u0 - omega * u1 */
    ierr = VecAXPBY( u1, alpha, omega, u0 ); CHKERRQ( ierr );
    ierr = VecAXPY( V0, -1.0, u1 ); CHKERRQ( ierr );

    /* compute velocity and incompressibility residual */
    E->monitor.vdotv = global_v_norm2_PETSc(E, V0);
    E->monitor.pdotp = global_p_norm2_PETSc(E, P0);
    v_norm = sqrt( E->monitor.vdotv );
    p_norm = sqrt( E->monitor.pdotp );
    dvelocity = sqrt( global_v_norm2_PETSc( E, u1 ) / 
                      (1e-32 + E->monitor.vdotv) );
    dpressure = sqrt( global_p_norm2_PETSc( E, st ) / 
                      (1e-32 + E->monitor.pdotp) );


    ierr = MatMult( E->DC, V0, t0 ); CHKERRQ( ierr );
    E->monitor.incompressibility = sqrt( global_div_norm2_PETSc( E, t0 )
                                        / (1e-32 + E->monitor.vdotv) );

    count++;

    if( E->control.print_convergence && E->parallel.me == 0 ) {
      print_convergence_progress( E, count, time0,
                                  v_norm, p_norm,
                                  dvelocity, dpressure,
                                  E->monitor.incompressibility );
    }

    /* shift array pointers */
    ierr = VecSwap( p1, p2 ); CHKERRQ( ierr );
    ierr = VecSwap( r1, r2 ); CHKERRQ( ierr );

    /* shift <r0, rt> = <r1, rt> */
    r0dotrt = r1dotrt;

    // recompute the norm of the residual
    ierr = VecNorm( r1, NORM_2, &r1_norm ); CHKERRQ( ierr );

  }

  // converged. now copy the converged values of V0 and P0 into V and P
  PetscReal *P_tmp, *V_tmp;
  ierr = VecGetArray( V0, &V_tmp ); CHKERRQ( ierr );
  for( m=1; m<=E->sphere.caps_per_proc; ++m ) {
    for( i = 0; i < neq; i++ )
      V[m][i+1] = V_tmp[i]; 
  }
  ierr = VecRestoreArray( V0, &V_tmp ); CHKERRQ( ierr );
  
  ierr = VecGetArray( P0, &P_tmp ); CHKERRQ( ierr );
  for( m = 1; m < E->sphere.caps_per_proc; ++m ) {
    for( i = 0; i < nel; i++ )
      P[1][i+1] = P_tmp[i]; 
  }
  ierr = VecRestoreArray( P0, &P_tmp ); CHKERRQ( ierr );


  ierr = VecDestroy( &rt ); CHKERRQ( ierr );
  ierr = VecDestroy( &p1 ); CHKERRQ( ierr );
  ierr = VecDestroy( &p2 ); CHKERRQ( ierr );
  ierr = VecDestroy( &BPI ); CHKERRQ( ierr );
  ierr = VecDestroy( &pt ); CHKERRQ( ierr );
  ierr = VecDestroy( &u0 ); CHKERRQ( ierr );
  ierr = VecDestroy( &s0 ); CHKERRQ( ierr );
  ierr = VecDestroy( &st ); CHKERRQ( ierr );
  ierr = VecDestroy( &u1 ); CHKERRQ( ierr );
  ierr = VecDestroy( &t0 ); CHKERRQ( ierr );
  ierr = VecDestroy( &r2 ); CHKERRQ( ierr );
  ierr = VecDestroy( &v0 ); CHKERRQ( ierr );
  ierr = VecDestroy( &V0 ); CHKERRQ( ierr );
  ierr = VecDestroy( &P0 ); CHKERRQ( ierr );

  *steps_max = count;

  PetscFunctionReturn(0);
}

/* Solve compressible Stokes flow using
 * bi-conjugate gradient stablized (BiCG-stab) iterations
 */

static void solve_Ahat_p_fhat_BiCG(struct All_variables *E,
                                   double **V, double **P, double **FF,
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

    double *F[NCS];
    double *r1[NCS], *r2[NCS], *pt[NCS], *p1[NCS], *p2[NCS];
    double *rt[NCS], *v0[NCS], *s0[NCS], *st[NCS], *t0[NCS];
    double *u0[NCS];
    double *shuffle[NCS];

    double time0, v_res;
    
    inner_imp = imp * E->control.inner_accuracy_scale; /* allow for different innner loop accuracy */

    npno = E->lmesh.npno;
    neq = E->lmesh.neq;
    lev = E->mesh.levmax;

    for (m=1; m<=E->sphere.caps_per_proc; m++)   {
        F[m] = (double *)malloc(neq*sizeof(double));
        r1[m] = (double *)malloc((npno+1)*sizeof(double));
        r2[m] = (double *)malloc((npno+1)*sizeof(double));
        pt[m] = (double *)malloc((npno+1)*sizeof(double));
        p1[m] = (double *)malloc((npno+1)*sizeof(double));
        p2[m] = (double *)malloc((npno+1)*sizeof(double));
        rt[m] = (double *)malloc((npno+1)*sizeof(double));
        v0[m] = (double *)malloc((npno+1)*sizeof(double));
        s0[m] = (double *)malloc((npno+1)*sizeof(double));
        st[m] = (double *)malloc((npno+1)*sizeof(double));
        t0[m] = (double *)malloc((npno+1)*sizeof(double));

        u0[m] = (double *)malloc(neq*sizeof(double));
    }

    time0 = CPU_time0();
    count = 0;
    v_res = E->monitor.fdotf;

    /* copy the original force vector since we need to keep it intact
       between iterations */
    for(m=1;m<=E->sphere.caps_per_proc;m++)
        for(j=0;j<neq;j++)
            F[m][j] = FF[m][j];


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
    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(j=1; j<=npno; j++)
            rt[m][j] = r1[m][j];


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
            for (m=1; m<=E->sphere.caps_per_proc; m++)
                for(j=1; j<=npno; j++)
                    p2[m][j] = r1[m][j];
        else {
            /* p2 = r1 + <r1,rt>/<r0,rt> * alpha/omega * (p1 - omega*v0) */
            beta = (r1dotrt / r0dotrt) * (alpha / omega);
            for(m=1; m<=E->sphere.caps_per_proc; m++)
                for(j=1; j<=npno; j++)
                    p2[m][j] = r1[m][j] + beta
                        * (p1[m][j] - omega * v0[m][j]);
        }


        /* preconditioner BPI ~= inv(K), pt = BPI*p2 */
        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=1; j<=npno; j++)
                pt[m][j] = E->BPI[lev][m][j] * p2[m][j];


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
        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=1; j<=npno; j++)
                s0[m][j] = r1[m][j] - alpha * v0[m][j];


        /* preconditioner BPI ~= inv(K), st = BPI*s0 */
        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=1; j<=npno; j++)
                st[m][j] = E->BPI[lev][m][j] * s0[m][j];


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
        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=1; j<=npno; j++)
                r2[m][j] = s0[m][j] - omega * t0[m][j];


        /* P = P + alpha * pt + omega * st */
        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=1; j<=npno; j++)
                s0[m][j] = alpha * pt[m][j] + omega * st[m][j];

        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=1; j<=npno; j++)
                P[m][j] += s0[m][j];


        /* V = V - alpha * u0 - omega * u1 */
        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=0; j<neq; j++)
                F[m][j] = alpha * u0[m][j] + omega * E->u1[m][j];

        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=0; j<neq; j++)
                V[m][j] -= F[m][j];


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
        for(m=1; m<=E->sphere.caps_per_proc; m++) {
            shuffle[m] = p1[m];
            p1[m] = p2[m];
            p2[m] = shuffle[m];

            shuffle[m] = r1[m];
            r1[m] = r2[m];
            r2[m] = shuffle[m];
        }

        /* shift <r0, rt> = <r1, rt> */
        r0dotrt = r1dotrt;

    } /* end loop for conjugate gradient */


    for(m=1; m<=E->sphere.caps_per_proc; m++) {
    	free((void *) F[m]);
        free((void *) r1[m]);
        free((void *) r2[m]);
        free((void *) pt[m]);
        free((void *) p1[m]);
        free((void *) p2[m]);
        free((void *) rt[m]);
        free((void *) v0[m]);
        free((void *) s0[m]);
        free((void *) st[m]);
        free((void *) t0[m]);

        free((void *) u0[m]);
    }

    *steps_max=count;

    return;

}


/* Solve compressible Stokes flow using
 * conjugate gradient (CG) iterations with an outer iteration
 */

static void solve_Ahat_p_fhat_iterCG(struct All_variables *E,
                                     double **V, double **P, double **F,
                                     double imp, int *steps_max)
{
    int m, i;
    int cycles, num_of_loop;
    double relative_err_v, relative_err_p;
    double *old_v[NCS], *old_p[NCS],*diff_v[NCS],*diff_p[NCS];
    double div_res;
    const int npno = E->lmesh.npno;
    const int neq = E->lmesh.neq;
    const int lev = E->mesh.levmax;

    double global_v_norm2(),global_p_norm2();
    double global_div_norm2();
    void assemble_div_rho_u();
    
    for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    	old_v[m] = (double *)malloc(neq*sizeof(double));
    	diff_v[m] = (double *)malloc(neq*sizeof(double));
    	old_p[m] = (double *)malloc((npno+1)*sizeof(double));
    	diff_p[m] = (double *)malloc((npno+1)*sizeof(double));
    }

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

        for (m=1;m<=E->sphere.caps_per_proc;m++) {
            for(i=0;i<neq;i++) old_v[m][i] = V[m][i];
            for(i=1;i<=npno;i++) old_p[m][i] = P[m][i];
        }

        if(E->control.use_petsc)
          solve_Ahat_p_fhat_CG_PETSc(E, V, P, F, imp, &cycles);
        else
          solve_Ahat_p_fhat_CG(E, V, P, F, imp, &cycles);


        /* compute norm of div(rho*V) */
        assemble_div_rho_u(E, V, E->u1, lev);
        div_res = sqrt(global_div_norm2(E, E->u1) / (1e-32 + E->monitor.vdotv));

        for (m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=0;i<neq;i++) diff_v[m][i] = V[m][i] - old_v[m][i];

        relative_err_v = sqrt( global_v_norm2(E,diff_v) /
                               (1.0e-32 + E->monitor.vdotv) );

        for (m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=npno;i++) diff_p[m][i] = P[m][i] - old_p[m][i];

        relative_err_p = sqrt( global_p_norm2(E,diff_p) /
                               (1.0e-32 + E->monitor.pdotp) );

        if(E->parallel.me == 0) {
            fprintf(stderr, "itercg -- div(rho*v)/v=%.2e dv/v=%.2e and dp/p=%.2e loop %d\n\n", div_res, relative_err_v, relative_err_p, num_of_loop);
            fprintf(E->fp, "itercg -- div(rho*v)/v=%.2e dv/v=%.2e and dp/p=%.2e loop %d\n\n", div_res, relative_err_v, relative_err_p, num_of_loop);
        }

        num_of_loop++;

    } /* end of while */

    for (m=1;m<=E->sphere.caps_per_proc;m++)   {
    	free((void *) old_v[m]);
    	free((void *) old_p[m]);
	free((void *) diff_v[m]);
	free((void *) diff_p[m]);
    }

    return;
}


static void initial_vel_residual(struct All_variables *E,
                                 double **V, double **P, double **F,
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
    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(i=0; i<neq; i++)
            F[m][i] = F[m][i] - E->u1[m][i];

    assemble_del2_u(E, V, E->u1, lev, 1);
    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(i=0; i<neq; i++)
            F[m][i] = F[m][i] - E->u1[m][i];

    strip_bcs_from_residual(E, F, lev);


    /* solve K*u1 = F for u1 */
    valid = solve_del2_u(E, E->u1, F, acc, lev);
    if(!valid && (E->parallel.me==0)) {
        fputs("Warning: solver not converging! 0\n", stderr);
        fputs("Warning: solver not converging! 0\n", E->fp);
    }
    strip_bcs_from_residual(E, E->u1, lev);


    /* V = V + u1 */
    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(i=0; i<neq; i++)
            V[m][i] += E->u1[m][i];

    return;
}
