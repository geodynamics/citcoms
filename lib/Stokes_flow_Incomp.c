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
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include <stdlib.h>

/* Master loop for pressure and (hence) velocity field */

void solve_constrained_flow_iterative(E)
     struct All_variables *E;

{
    float solve_Ahat_p_fhat();
    void v_from_vector();
    void p_to_nodes();

    int cycles;

    cycles=E->control.p_iterations;

    /* Solve for velocity and pressure, correct for bc's */

    solve_Ahat_p_fhat(E,E->U,E->P,E->F,E->control.accuracy,&cycles);

    v_from_vector(E);
    p_to_nodes(E,E->P,E->NP,E->mesh.levmax);

    return;
}

void solve_constrained_flow_iterative_pseudo_surf(E)
     struct All_variables *E;

{
    float solve_Ahat_p_fhat();
    void v_from_vector_pseudo_surf();
    void p_to_nodes();

    int cycles;

    cycles=E->control.p_iterations;

    /* Solve for velocity and pressure, correct for bc's */

    solve_Ahat_p_fhat(E,E->U,E->P,E->F,E->control.accuracy,&cycles);

    v_from_vector_pseudo_surf(E);
    p_to_nodes(E,E->P,E->NP,E->mesh.levmax);

    return;
}


/* ========================================================================= */

float solve_Ahat_p_fhat(struct All_variables *E,
                        double **V, double **P, double **FF,
                        double imp, int *steps_max)
{
    int m, i, j, count, valid, lev, npno, neq;
    int gnpno, gneq;

    double *r1[NCS], *F[NCS];
    double *r0[NCS], *r2[NCS], *z0[NCS], *z1[NCS], *s1[NCS], *s2[NCS];
    double *shuffle[NCS];
    double alpha, delta, s2dotAhat, r0dotr0, r1dotz1;
    double residual, v_res;

    double global_vdot(), global_pdot();
    double *dvector();

    double time0, CPU_time0();
    float dpressure, dvelocity;

    void assemble_div_u();
    void assemble_del2_u();
    void assemble_grad_p();
    void strip_bcs_from_residual();
    int  solve_del2_u();
    void parallel_process_termination();

    gnpno = E->mesh.npno;
    gneq = E->mesh.neq;
    npno = E->lmesh.npno;
    neq = E->lmesh.neq;

    for (m=1; m<=E->sphere.caps_per_proc; m++) {
        F[m] = (double *)malloc((neq+1)*sizeof(double));

        r0[m] = (double *)malloc((npno+1)*sizeof(double));
        r1[m] = (double *)malloc((npno+1)*sizeof(double));
        r2[m] = (double *)malloc((npno+1)*sizeof(double));
        z0[m] = (double *)malloc((npno+1)*sizeof(double));
        z1[m] = (double *)malloc((npno+1)*sizeof(double));
        s1[m] = (double *)malloc((npno+1)*sizeof(double));
        s2[m] = (double *)malloc((npno+1)*sizeof(double));
    }

    /* Copy the original force vector. FF shouldn't be modified. */
    for (m=1;m<=E->sphere.caps_per_proc;m++)
        for(i=0;i<neq;i++)
            F[m][i] = FF[m][i];

    time0 = CPU_time0();


    /* calculate the initial velocity residual */
    lev = E->mesh.levmax;
    v_res = sqrt(global_vdot(E, F, F, lev)/gneq);

    if (E->parallel.me==0) {
        fprintf(E->fp, "initial residue of momentum equation F %.9e %d\n",
                v_res, gneq);
        fprintf(stderr, "initial residue of momentum equation F %.9e %d\n",
                v_res, gneq);
    }


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
    valid=solve_del2_u(E, E->u1, F, imp*v_res, E->mesh.levmax);
    strip_bcs_from_residual(E, E->u1, lev);


    /* V = V + u1 */
    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(i=0; i<neq; i++)
            V[m][i] += E->u1[m][i];


    /* r1 = div(V) */
    assemble_div_u(E, V, r1, lev);

    /* incompressiblity residual = norm(r1) / norm(V) */
    residual = sqrt(global_pdot(E, r1, r1, lev)/gnpno);
    E->monitor.vdotv = sqrt(global_vdot(E, V, V, lev)/gneq);
    E->monitor.incompressibility = residual / E->monitor.vdotv;

    count = 0;

    if (E->control.print_convergence && E->parallel.me==0) {
        fprintf(E->fp, "AhatP (%03d) after %g seconds with div/v=%.3e "
                "for step %d\n", count, CPU_time0()-time0,
                E->monitor.incompressibility, E->monitor.solution_cycles);
        fprintf(stderr, "AhatP (%03d) after %g seconds with div/v=%.3e "
                "for step %d\n", count, CPU_time0()-time0,
                E->monitor.incompressibility, E->monitor.solution_cycles);
    }


    /* pressure and velocity corrections */
    dpressure = 1.0;
    dvelocity = 1.0;

    while( (valid) && (count < *steps_max) &&
           (E->monitor.incompressibility >= E->control.tole_comp) &&
           (dpressure >= imp) && (dvelocity >= imp) )  {


        /* preconditioner B, z1 = B*r1 */
        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=1; j<=npno; j++)
                z1[m][j] = E->BPI[lev][m][j] * r1[m][j];


        /* r1dotz1 = <r1, z1> */
        r1dotz1 = global_pdot(E, r1, z1, lev);


        if ((count == 0))
            for (m=1; m<=E->sphere.caps_per_proc; m++)
                for(j=1; j<=npno; j++)
                    s2[m][j] = z1[m][j];
        else {
            /* s2 = z1 + s1 * <r1,z1>/<r0,z0> */
            r0dotr0 = global_pdot(E, r0, z0, lev);
            assert(r0dotr0 != 0.0  /* Division by zero in head of incompressibility iteration */);
            delta = r1dotz1 / r0dotr0;
            for(m=1; m<=E->sphere.caps_per_proc; m++)
                for(j=1; j<=npno; j++)
                    s2[m][j] = z1[m][j] + delta * s1[m][j];
        }

        /* solve K*u1 = grad(s2) for u1 */
        assemble_grad_p(E, s2, F, lev);
        valid = solve_del2_u(E, E->u1, F, imp*v_res, lev);
        strip_bcs_from_residual(E, E->u1, lev);


        /* alpha = <r1, z1> / <s2, div(u1)> */
        assemble_div_u(E, E->u1, F, lev);
        s2dotAhat = global_pdot(E, s2, F, lev);

        if(valid)
            /* alpha defined this way is the same as R&W */
            alpha = r1dotz1 / s2dotAhat;
        else
            alpha = 0.0;


        /* r2 = r1 - alpha * div(u1) */
        /* P = P + alpha * s2 */
        /* V = V - alpha * u1 */
        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=1; j<=npno; j++)   {
                r2[m][j] = r1[m][j] - alpha * F[m][j];
                P[m][j] += alpha * s2[m][j];
            }

        for(m=1; m<=E->sphere.caps_per_proc; m++)
            for(j=0; j<neq; j++)
                V[m][j] -= alpha * E->u1[m][j];


        /* compute velocity and incompressibility residual */
        assemble_div_u(E, V, F, lev);
        E->monitor.vdotv = global_vdot(E, V, V, E->mesh.levmax);
        E->monitor.incompressibility = sqrt((gneq/gnpno)
                                            *(1.0e-32
                                              + global_pdot(E, F, F, lev)
                                              / (1.0e-32+E->monitor.vdotv)));


        /* compute velocity and pressure corrections */
        dpressure = alpha * sqrt(global_pdot(E, s2, s2, lev)
                                 / (1.0e-32 + global_pdot(E, P, P, lev)));
        dvelocity = alpha * sqrt(global_vdot(E, E->u1, E->u1, lev)
                                 / (1.0e-32 + E->monitor.vdotv));

        count++;

        if(E->control.print_convergence && E->parallel.me==0) {
            fprintf(E->fp, "AhatP (%03d) after %g seconds with div/v=%.3e "
                    "dv/v=%.3e and dp/p=%.3e for step %d\n",
                    count, CPU_time0()-time0, E->monitor.incompressibility,
                    dvelocity, dpressure, E->monitor.solution_cycles);
            fprintf(stderr, "AhatP (%03d) after %g seconds with div/v=%.3e "
                    "dv/v=%.3e and dp/p=%.3e for step %d\n",
                    count, CPU_time0()-time0, E->monitor.incompressibility,
                    dvelocity, dpressure, E->monitor.solution_cycles);
        }


        /* swap array pointers */
        for(m=1; m<=E->sphere.caps_per_proc; m++) {
            shuffle[m] = s1[m];
            s1[m] = s2[m];
            s2[m] = shuffle[m];

            shuffle[m] = r0[m];
            r0[m] = r1[m];
            r1[m] = r2[m];
            r2[m] = shuffle[m];

            shuffle[m] = z0[m];
            z0[m] = z1[m];
            z1[m] = shuffle[m];
        }

    }       /* end loop for conjugate gradient   */

    for(m=1; m<=E->sphere.caps_per_proc; m++) {
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




