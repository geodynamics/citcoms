/* -*- C -*- */
/* vim:set ft=c: */

#include <math.h>
#include "global_defs.h"


#ifndef __CUDACC__
#define __device__
#define __global__
#define __host__
#else
/* XXX */
#undef assert
#define assert(c)
#endif


/*------------------------------------------------------------------------*/
/* from element_definitions.h */

__device__ static const int enodes[] = {0,2,4,8};
__device__ static const int loc_mat_size[] = {0,4,8,24};


/*------------------------------------------------------------------------*/
/* from Regional_parallel_related.c */

__device__ void regional_exchange_id_d(
    struct All_variables *E,
    double **U,
    int lev,
    double *memory
    )
{

    int j,m,k;
    double *S[27],*R[27];
    int dimofk;
    
#if 0 /* XXX */
    MPI_Status status;
#endif
    
    for (m=1;m<=E->sphere.caps_per_proc;m++)    {
        for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {
            dimofk = 1+E->parallel.NUM_NEQ[lev][m].pass[k];
            S[k] = memory; memory += dimofk;
            R[k] = memory; memory += dimofk;
        }
    }

    for (m=1;m<=E->sphere.caps_per_proc;m++)   {
        for (k=1;k<=E->parallel.TNUM_PASS[lev][m];k++)  {

            for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[k];j++)
                S[k][j-1] = U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[k] ];
     
#if 0 /* XXX */
            MPI_Sendrecv(S[k],E->parallel.NUM_NEQ[lev][m].pass[k],MPI_DOUBLE,
                         E->parallel.PROCESSOR[lev][m].pass[k],1,
                         R[k],E->parallel.NUM_NEQ[lev][m].pass[k],MPI_DOUBLE,
                         E->parallel.PROCESSOR[lev][m].pass[k],1,
                         E->parallel.world,&status);
#endif

            for (j=1;j<=E->parallel.NUM_NEQ[lev][m].pass[k];j++)
                U[m][ E->parallel.EXCHANGE_ID[lev][m][j].pass[k] ] += R[k][j-1];

        }           /* for k */
    }     /* for m */         /* finish sending */
    
    return;
}


/*------------------------------------------------------------------------*/
/* from Full_parallel_related.c */

__device__ void full_exchange_id_d(
    struct All_variables *E,
    double **U,
    int lev,
    double *memory
    )
{
    /* XXX */
}


/*------------------------------------------------------------------------*/
/* from BC_util.c */

__device__ void strip_bcs_from_residual(
    struct All_variables *E,
    double **Res,
    int level
    )
{
    int m,i;

    for (m=1;m<=E->sphere.caps_per_proc;m++)
        if (E->num_zero_resid[level][m])
            for(i=1;i<=E->num_zero_resid[level][m];i++)
                Res[m][E->zero_resid[level][m][i]] = 0.0;

    return;
}


/*------------------------------------------------------------------------*/
/* from Element_calculations.c */

__device__ void e_assemble_del2_u(
    struct All_variables *E,
    double **u, double **Au,
    int level,
    int strip_bcs,
    double *memory
    )
{
    int  e,i,a,b,a1,a2,a3,ii,m,nodeb;

    const int n=loc_mat_size[E->mesh.nsd];
    const int ends=enodes[E->mesh.nsd];
    const int dims=E->mesh.nsd;
    const int nel=E->lmesh.NEL[level];
    const int neq=E->lmesh.NEQ[level];

    for (m=1;m<=E->sphere.caps_per_proc;m++)   {
        for(i=0;i<neq;i++)
            Au[m][i] = 0.0;

        for(e=1;e<=nel;e++)   {
            for(a=1;a<=ends;a++) {
                ii = E->IEN[level][m][e].node[a];
                a1 = E->ID[level][m][ii].doff[1];
                a2 = E->ID[level][m][ii].doff[2];
                a3 = E->ID[level][m][ii].doff[3];
                for(b=1;b<=ends;b++) {
                    nodeb = E->IEN[level][m][e].node[b];
                    ii = (a*n+b)*dims-(dims*n+dims);
                    /* i=1, j=1,2 */
                    /* i=1, j=1,2,3 */
                    Au[m][a1] +=
                        E->elt_k[level][m][e].k[ii] *
                        u[m][E->ID[level][m][nodeb].doff[1]]
                        + E->elt_k[level][m][e].k[ii+1] *
                        u[m][E->ID[level][m][nodeb].doff[2]]
                        + E->elt_k[level][m][e].k[ii+2] *
                        u[m][E->ID[level][m][nodeb].doff[3]];
                    /* i=2, j=1,2,3 */
                    Au[m][a2] +=
                        E->elt_k[level][m][e].k[ii+n] *
                        u[m][E->ID[level][m][nodeb].doff[1]]
                        + E->elt_k[level][m][e].k[ii+n+1] *
                        u[m][E->ID[level][m][nodeb].doff[2]]
                        + E->elt_k[level][m][e].k[ii+n+2] *
                        u[m][E->ID[level][m][nodeb].doff[3]];
                    /* i=3, j=1,2,3 */
                    Au[m][a3] +=
                        E->elt_k[level][m][e].k[ii+n+n] *
                        u[m][E->ID[level][m][nodeb].doff[1]]
                        + E->elt_k[level][m][e].k[ii+n+n+1] *
                        u[m][E->ID[level][m][nodeb].doff[2]]
                        + E->elt_k[level][m][e].k[ii+n+n+2] *
                        u[m][E->ID[level][m][nodeb].doff[3]];

                }         /* end for loop b */
            }             /* end for loop a */

        }          /* end for e */
    }         /* end for m  */
    
    if (0) { /* XXX */
        full_exchange_id_d(E, Au, level, memory);
    } else {
        regional_exchange_id_d(E, Au, level, memory);
    }

    if(strip_bcs)
        strip_bcs_from_residual(E,Au,level);

    return;
}

__device__ void n_assemble_del2_u(
    struct All_variables *E,
    double **u, double **Au,
    int level,
    int strip_bcs,
    double *memory
    )
{
    int m, e,i;
    int eqn1,eqn2,eqn3;

    double UU,U1,U2,U3;

    int *C;
    higher_precision *B1,*B2,*B3;

    const int neq=E->lmesh.NEQ[level];
    const int nno=E->lmesh.NNO[level];
    const int dims=E->mesh.nsd;
    const int max_eqn = dims*14;


    for (m=1;m<=E->sphere.caps_per_proc;m++)  {

        for(e=0;e<=neq;e++)
            Au[m][e]=0.0;

        u[m][neq] = 0.0;

        for(e=1;e<=nno;e++)     {

            eqn1=E->ID[level][m][e].doff[1];
            eqn2=E->ID[level][m][e].doff[2];
            eqn3=E->ID[level][m][e].doff[3];

            U1 = u[m][eqn1];
            U2 = u[m][eqn2];
            U3 = u[m][eqn3];

            C=E->Node_map[level][m] + (e-1)*max_eqn;
            B1=E->Eqn_k1[level][m]+(e-1)*max_eqn;
            B2=E->Eqn_k2[level][m]+(e-1)*max_eqn;
            B3=E->Eqn_k3[level][m]+(e-1)*max_eqn;

            for(i=3;i<max_eqn;i++)  {
                UU = u[m][C[i]];
                Au[m][eqn1] += B1[i]*UU;
                Au[m][eqn2] += B2[i]*UU;
                Au[m][eqn3] += B3[i]*UU;
            }
            for(i=0;i<max_eqn;i++)
                Au[m][C[i]] += B1[i]*U1+B2[i]*U2+B3[i]*U3;

        }     /* end for e */
    }     /* end for m */
    
    if (0) { /* XXX */
        full_exchange_id_d(E, Au, level, memory);
    } else {
        regional_exchange_id_d(E, Au, level, memory);
    }

    if (strip_bcs)
        strip_bcs_from_residual(E,Au,level);

    return;
}

__device__ void assemble_del2_u(
    struct All_variables *E,
    double **u, double **Au,
    int level,
    int strip_bcs,
    double *memory
    )
{
    if(E->control.NMULTIGRID||E->control.NASSEMBLE)
        n_assemble_del2_u(E,u,Au,level,strip_bcs, memory);
    else
        e_assemble_del2_u(E,u,Au,level,strip_bcs, memory);

    return;
}


/*------------------------------------------------------------------------*/
/* from Global_operations.c */

__device__ double global_vdot(
    struct All_variables *E,
    double **A, double **B,
    int lev
    )
{
    int m,i,neq;
    double prod, temp,temp1;

    temp = 0.0;
    prod = 0.0;

    for (m=1;m<=E->sphere.caps_per_proc;m++)  {
        neq=E->lmesh.NEQ[lev];
        temp1 = 0.0;
        for (i=0;i<neq;i++)
            temp += A[m][i]*B[m][i];

        for (i=1;i<=E->parallel.Skip_neq[lev][m];i++)
            temp1 += A[m][E->parallel.Skip_id[lev][m][i]]*B[m][E->parallel.Skip_id[lev][m][i]];

        temp -= temp1;

    }
  
#if 0 /* XXX */
    MPI_Allreduce(&temp, &prod,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
#endif

    return (prod);
}


/*------------------------------------------------------------------------*/
/* from General_matix_functions.c */

__device__ double conj_grad(
    struct All_variables *E,
    double **d0,
    double **F,
    double acc,
    int *cycles,
    int level,
    double *memory
    )
{
    double *r0[NCS],*r1[NCS],*r2[NCS];
    double *z0[NCS],*z1[NCS];
    double *p1[NCS],*p2[NCS];
    double *Ap[NCS];
    double *shuffle[NCS];

    int m,count,i,steps;
    double residual;
    double alpha,beta,dotprod,dotr1z1,dotr0z0;

    const int mem_lev=E->mesh.levmax;
    const int high_neq = E->lmesh.NEQ[level];
    
    steps = *cycles;
    
    for(m=1;m<=E->sphere.caps_per_proc;m++)    {
        r0[m] = memory; memory += E->lmesh.NEQ[mem_lev];
        r1[m] = memory; memory += E->lmesh.NEQ[mem_lev];
        r2[m] = memory; memory += E->lmesh.NEQ[mem_lev];
        z0[m] = memory; memory += E->lmesh.NEQ[mem_lev];
        z1[m] = memory; memory += E->lmesh.NEQ[mem_lev];
        p1[m] = memory; memory += (1+E->lmesh.NEQ[mem_lev]);
        p2[m] = memory; memory += (1+E->lmesh.NEQ[mem_lev]);
        Ap[m] = memory; memory += (1+E->lmesh.NEQ[mem_lev]);
    }

    for(m=1;m<=E->sphere.caps_per_proc;m++)
        for(i=0;i<high_neq;i++) {
            r1[m][i] = F[m][i];
            d0[m][i] = 0.0;
        }

    residual = sqrt(global_vdot(E,r1,r1,level));

    assert(residual != 0.0  /* initial residual for CG = 0.0 */);
    count = 0;

    while (((residual > acc) && (count < steps)) || count == 0)  {

        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=0;i<high_neq;i++)
                z1[m][i] = E->BI[level][m][i] * r1[m][i];

        dotr1z1 = global_vdot(E,r1,z1,level);

        if (0==count)
            for(m=1;m<=E->sphere.caps_per_proc;m++)
                for(i=0;i<high_neq;i++)
                    p2[m][i] = z1[m][i];
        else {
            assert(dotr0z0 != 0.0 /* in head of conj_grad */);
            beta = dotr1z1/dotr0z0;
            for(m=1;m<=E->sphere.caps_per_proc;m++)
                for(i=0;i<high_neq;i++)
                    p2[m][i] = z1[m][i] + beta * p1[m][i];
        }

        dotr0z0 = dotr1z1;

        assemble_del2_u(E,p2,Ap,level,1,memory);

        dotprod=global_vdot(E,p2,Ap,level);

        if(0.0==dotprod)
            alpha=1.0e-3;
        else
            alpha = dotr1z1/dotprod;

        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=0;i<high_neq;i++) {
                d0[m][i] += alpha * p2[m][i];
                r2[m][i] = r1[m][i] - alpha * Ap[m][i];
            }

        residual = sqrt(global_vdot(E,r2,r2,level));

        for(m=1;m<=E->sphere.caps_per_proc;m++)    {
            shuffle[m] = r0[m]; r0[m] = r1[m]; r1[m] = r2[m]; r2[m] = shuffle[m];
            shuffle[m] = z0[m]; z0[m] = z1[m]; z1[m] = shuffle[m];
            shuffle[m] = p1[m]; p1[m] = p2[m]; p2[m] = shuffle[m];
        }

        count++;
        /* end of while-loop */

    }

    *cycles=count;

    strip_bcs_from_residual(E,d0,level);
    
    return(residual);
}


__global__ void solve_del2_u(
    struct All_variables *E,
    double **d0,
    double **F,
    double acc,
    int high_lev,
    double *memory,
    int *valid
    )
{
    int count,cycles;
    int i, neq, m;

    double residual;

    neq  = E->lmesh.NEQ[high_lev];

    for (m=1;m<=E->sphere.caps_per_proc;m++)
        for(i=0;i<neq;i++)  {
            d0[m][i] = 0.0;
        }

    residual = sqrt(global_vdot(E,F,F,high_lev));

    count = 0;

    assert(!E->control.NMULTIGRID);
    /* conjugate gradient solution */

    cycles = E->control.v_steps_low;
    residual = conj_grad(E,d0,F,acc,&cycles,high_lev,memory);
    *valid = (residual < acc)? 1:0;

    count++;

    E->monitor.momentum_residual = residual;
    E->control.total_iteration_cycles += count;
    E->control.total_v_solver_calls += 1;

    return;
}


/*------------------------------------------------------------------------*/

__host__ int main() {
#ifndef __CUDACC__
    if (0) {
        struct All_variables *E;
        const int levmax = E->mesh.levmax;
        int m,k;
        
        double *memory;
        int memoryDim;
        int valid;

        memoryDim = 0;
        /* conj_grad */
        memoryDim += E->sphere.caps_per_proc *
                     (5 * E->lmesh.NEQ[levmax]  + /* r0,r1,r2,z0,z1 */
                      3 * (1+E->lmesh.NEQ[levmax]) /* p1,p2,Ap */
                         );
        /* regional_exchange_id_d */
        for (m=1;m<=E->sphere.caps_per_proc;m++)    {
            for (k=1;k<=E->parallel.TNUM_PASS[levmax][m];k++)  {
                memoryDim += 2 * (1+E->parallel.NUM_NEQ[levmax][m].pass[k]); /* S,R */
            }
        }
        
        memory = (double *)malloc(memoryDim*sizeof(double));
        
        solve_del2_u(0,0,0,0,0,memory,&valid);
        
        free(memory);
    }
#endif
    return 0;
}
