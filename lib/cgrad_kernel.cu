/* -*- C -*- */
/* vim:set ft=c: */

#include <math.h>
#include "global_defs.h"


#ifdef __CUDACC__
#undef assert
#define assert(c)
#else
#define __device__ static inline
#define __global__ static
#define __host__
#endif


enum {
    LEVEL = 0,
    CAPS_PER_PROC = 1,
    M = 1, /* cap # */
    NSD = 3, /* Spatial extent: 3d */
    MAX_EQN = NSD*14,
    ENODES = 8, /* enodes[NSD] */
    LOC_MAT_SIZE = 24, /* loc_mat_size[NSD] */
};


struct octoterm {
    /* an octoterm is 8 (ENODES) * 3 terms */
    int e;
    int a;
    int offset;
};


struct matrix_mult {
    int n; /* number of octoterms: 1, 2, 4, or 8 */
    struct octoterm *ot;
};


struct Some_variables {
    int num_zero_resid;
    int *zero_resid;
    
    struct /*MESH_DATA*/ {
        int NEQ;
        int NNO;
        int NEL;
    } lmesh;
    
    struct IEN *IEN;
    struct ID *ID;
    struct EK *elt_k;
    
    higher_precision *Eqn_k1, *Eqn_k2, *Eqn_k3;
    int *Node_map;
    
    double *BI;
    
    struct /*CONTROL*/ {
        int NASSEMBLE;
        
        int v_steps_low;
        int total_iteration_cycles;
        int total_v_solver_calls;
    } control;
    
    /* temporary malloc'd memory */
    double *memory;
    int memoryDim;
    
    struct matrix_mult *mm; /* dim is 'neq' */
    
    /* outputs */
    
    struct /*MONITOR*/ {
        double momentum_residual;
    } monitor;

    int valid;
};


/*------------------------------------------------------------------------*/
/* from BC_util.c */

__device__ void strip_bcs_from_residual(
    struct Some_variables *E,
    double *Res
    )
{
    int i;

    for(i=1;i<=E->num_zero_resid;i++)
        Res[E->zero_resid[i]] = 0.0;

    return;
}


/*------------------------------------------------------------------------*/
/* from Element_calculations.c */

static int e_tally(
    struct Some_variables *E
    )
{
    int  e,i,a,a1,a2,a3,ii,nTotal;

    const int nel=E->lmesh.NEL;
    const int neq=E->lmesh.NEQ;
    
    nTotal = 0;
    for(i=0;i<neq;i++) {
        E->mm[i].n = 0;
    }

    for(e=1;e<=nel;e++)   {
        
        for(a=1;a<=ENODES;a++) {
            
            ii = E->IEN[e].node[a];
            a1 = E->ID[ii].doff[1];
            a2 = E->ID[ii].doff[2];
            a3 = E->ID[ii].doff[3];
            
            ++E->mm[a1].n;
            ++E->mm[a2].n;
            ++E->mm[a3].n;
            nTotal += 3;
            
        }             /* end for loop a */

    }          /* end for e */
    
    /* return the total number of octoterms */
    return nTotal;
}

static void e_map(
    struct Some_variables *E,
    struct octoterm *ot
    )
{
    int  e,i,a,a1,a2,a3,o1,o2,o3,ii;

    const int nel=E->lmesh.NEL;
    const int neq=E->lmesh.NEQ;

    for(i=0;i<neq;i++) {
        E->mm[i].ot = ot;
        ot += E->mm[i].n;
        E->mm[i].n = 0;
    }

    for(e=1;e<=nel;e++)   {
        
        for(a=1;a<=ENODES;a++) {
            
            ii = E->IEN[e].node[a];
            a1 = E->ID[ii].doff[1];
            a2 = E->ID[ii].doff[2];
            a3 = E->ID[ii].doff[3];
            
            o1 = E->mm[a1].n++;
            o2 = E->mm[a2].n++;
            o3 = E->mm[a3].n++;
            
            E->mm[a1].ot[o1].e = e;
            E->mm[a1].ot[o1].a = a;
            E->mm[a1].ot[o1].offset = 0;
            
            E->mm[a2].ot[o2].e = e;
            E->mm[a2].ot[o2].a = a;
            E->mm[a2].ot[o2].offset = LOC_MAT_SIZE;
            
            E->mm[a3].ot[o3].e = e;
            E->mm[a3].ot[o3].a = a;
            E->mm[a3].ot[o3].offset = LOC_MAT_SIZE + LOC_MAT_SIZE;
            
        }             /* end for loop a */

    }          /* end for e */
    
    return;
}

__device__ void dp_e_assemble_del2_u(
    struct Some_variables *E,
    double *u, double *Au,
    int strip_bcs
    )
{
    int  e,i,a,b,o,ii,nodeb,offset;
    double sum;

    const int neq=E->lmesh.NEQ;
    
    for (i = 0; i < neq; i++) {
        
        sum = 0.0;
        
        /* ENODES*ENODES = 8*8 = 64 threads per block */
        /* XXX: 8*(8-n) wasted threads */
        
        for (o = 0; o < E->mm[i].n; o++) {
            
            e      = E->mm[i].ot[o].e;
            a      = E->mm[i].ot[o].a;
            offset = E->mm[i].ot[o].offset;
            
            for (b = 1; b <= ENODES; b++) {
                
                /* each thread computes three terms */

                nodeb = E->IEN[e].node[b];
                ii = (a*LOC_MAT_SIZE+b)*NSD-(NSD*LOC_MAT_SIZE+NSD);
                
                /* XXX: must reduce here */
                sum +=
                    E->elt_k[e].k[ii+offset] *
                    u[E->ID[nodeb].doff[1]]
                    + E->elt_k[e].k[ii+offset+1] *
                    u[E->ID[nodeb].doff[2]]
                    + E->elt_k[e].k[ii+offset+2] *
                    u[E->ID[nodeb].doff[3]];
                
            }
        }
        
        /* each block writes one element, Au[i] */
        Au[i] = sum;
    }

    if (strip_bcs)
        strip_bcs_from_residual(E,Au);

    return;
}

__device__ void e_assemble_del2_u(
    struct Some_variables *E,
    double *u, double *Au,
    int strip_bcs
    )
{
    int  e,i,a,b,a1,a2,a3,ii,nodeb;

    const int nel=E->lmesh.NEL;
    const int neq=E->lmesh.NEQ;

    for(i=0;i<neq;i++)
        Au[i] = 0.0;

    for(e=1;e<=nel;e++)   {
        for(a=1;a<=ENODES;a++) {
            ii = E->IEN[e].node[a];
            a1 = E->ID[ii].doff[1];
            a2 = E->ID[ii].doff[2];
            a3 = E->ID[ii].doff[3];
            for(b=1;b<=ENODES;b++) {
                nodeb = E->IEN[e].node[b];
                ii = (a*LOC_MAT_SIZE+b)*NSD-(NSD*LOC_MAT_SIZE+NSD);
                /* i=1, j=1,2 */
                /* i=1, j=1,2,3 */
                Au[a1] +=
                    E->elt_k[e].k[ii] *
                    u[E->ID[nodeb].doff[1]]
                    + E->elt_k[e].k[ii+1] *
                    u[E->ID[nodeb].doff[2]]
                    + E->elt_k[e].k[ii+2] *
                    u[E->ID[nodeb].doff[3]];
                /* i=2, j=1,2,3 */
                Au[a2] +=
                    E->elt_k[e].k[ii+LOC_MAT_SIZE] *
                    u[E->ID[nodeb].doff[1]]
                    + E->elt_k[e].k[ii+LOC_MAT_SIZE+1] *
                    u[E->ID[nodeb].doff[2]]
                    + E->elt_k[e].k[ii+LOC_MAT_SIZE+2] *
                    u[E->ID[nodeb].doff[3]];
                /* i=3, j=1,2,3 */
                Au[a3] +=
                    E->elt_k[e].k[ii+LOC_MAT_SIZE+LOC_MAT_SIZE] *
                    u[E->ID[nodeb].doff[1]]
                    + E->elt_k[e].k[ii+LOC_MAT_SIZE+LOC_MAT_SIZE+1] *
                    u[E->ID[nodeb].doff[2]]
                    + E->elt_k[e].k[ii+LOC_MAT_SIZE+LOC_MAT_SIZE+2] *
                    u[E->ID[nodeb].doff[3]];

            }         /* end for loop b */
        }             /* end for loop a */

    }          /* end for e */
    
    if(strip_bcs)
        strip_bcs_from_residual(E,Au);

    return;
}

__device__ void n_assemble_del2_u(
    struct Some_variables *E,
    double *u, double *Au,
    int strip_bcs
    )
{
    int e,i;
    int eqn1,eqn2,eqn3;

    double UU,U1,U2,U3;

    int *C;
    higher_precision *B1,*B2,*B3;

    const int neq=E->lmesh.NEQ;
    const int nno=E->lmesh.NNO;
    
    /*
     * Au = E->Eqn_k? * u
     *  where E->Eqn_k? is the sparse stiffness matrix
     */

    for(e=0;e<=neq;e++)
        Au[e]=0.0;

    u[neq] = 0.0;

    for(e=1;e<=nno;e++)     {

        eqn1=E->ID[e].doff[1];
        eqn2=E->ID[e].doff[2];
        eqn3=E->ID[e].doff[3];

        U1 = u[eqn1];
        U2 = u[eqn2];
        U3 = u[eqn3];

        C=E->Node_map + (e-1)*MAX_EQN;
        B1=E->Eqn_k1+(e-1)*MAX_EQN;
        B2=E->Eqn_k2+(e-1)*MAX_EQN;
        B3=E->Eqn_k3+(e-1)*MAX_EQN;

        for(i=3;i<MAX_EQN;i++)  {
            UU = u[C[i]];
            Au[eqn1] += B1[i]*UU;
            Au[eqn2] += B2[i]*UU;
            Au[eqn3] += B3[i]*UU;
        }
        for(i=0;i<MAX_EQN;i++)
            Au[C[i]] += B1[i]*U1+B2[i]*U2+B3[i]*U3;

    }     /* end for e */
    
    if (strip_bcs)
        strip_bcs_from_residual(E,Au);

    return;
}

__device__ void assemble_del2_u(
    struct Some_variables *E,
    double *u, double *Au,
    int strip_bcs
    )
{
    if (E->control.NASSEMBLE)
        n_assemble_del2_u(E,u,Au,strip_bcs);
    else if (1)
        dp_e_assemble_del2_u(E,u,Au,strip_bcs);
    else
        e_assemble_del2_u(E,u,Au,strip_bcs);

    return;
}


/*------------------------------------------------------------------------*/
/* from Global_operations.c */

__device__ double global_vdot(
    struct Some_variables *E,
    double *A, double *B
    )
{
    int i,neq;
    double prod;

    prod = 0.0;

    neq=E->lmesh.NEQ;
    for (i=0;i<neq;i++)
        prod += A[i]*B[i];

    return prod;
}


/*------------------------------------------------------------------------*/
/* from General_matix_functions.c */

__device__ double conj_grad(
    struct Some_variables *E,
    double *d0,
    double *F,
    double acc,
    int *cycles
    )
{
    double *r0,*r1,*r2;
    double *z0,*z1;
    double *p1,*p2;
    double *Ap;
    double *shuffle;

    int count,i,steps;
    double residual;
    double alpha,beta,dotprod,dotr1z1,dotr0z0;
    
    double *memory = E->memory;

    const int neq = E->lmesh.NEQ;
    
    steps = *cycles;
    
    r0 = memory; memory += neq;
    r1 = memory; memory += neq;
    r2 = memory; memory += neq;
    z0 = memory; memory += neq;
    z1 = memory; memory += neq;
    p1 = memory; memory += (1+neq);
    p2 = memory; memory += (1+neq);
    Ap = memory; memory += (1+neq);
    assert(memory == E->memory + E->memoryDim);

    for(i=0;i<neq;i++) {
        r1[i] = F[i];
        d0[i] = 0.0;
    }

    residual = sqrt(global_vdot(E,r1,r1));

    assert(residual != 0.0  /* initial residual for CG = 0.0 */);
    count = 0;

    while (((residual > acc) && (count < steps)) || count == 0)  {

        for(i=0;i<neq;i++)
            z1[i] = E->BI[i] * r1[i];

        dotr1z1 = global_vdot(E,r1,z1);

        if (0==count)
            for(i=0;i<neq;i++)
                p2[i] = z1[i];
        else {
            assert(dotr0z0 != 0.0 /* in head of conj_grad */);
            beta = dotr1z1/dotr0z0;
            for(i=0;i<neq;i++)
                p2[i] = z1[i] + beta * p1[i];
        }

        dotr0z0 = dotr1z1;

        assemble_del2_u(E,p2,Ap,1);

        dotprod=global_vdot(E,p2,Ap);

        if(0.0==dotprod)
            alpha=1.0e-3;
        else
            alpha = dotr1z1/dotprod;

        for(i=0;i<neq;i++) {
            d0[i] += alpha * p2[i];
            r2[i] = r1[i] - alpha * Ap[i];
        }

        residual = sqrt(global_vdot(E,r2,r2));

        shuffle = r0; r0 = r1; r1 = r2; r2 = shuffle;
        shuffle = z0; z0 = z1; z1 = shuffle;
        shuffle = p1; p1 = p2; p2 = shuffle;

        count++;
        /* end of while-loop */

    }

    *cycles=count;

    strip_bcs_from_residual(E,d0);
    
    return residual;
}


__global__ void solve_del2_u(
    struct Some_variables *E,
    double *d0,
    double *F,
    double acc
    )
{
    int count,cycles;
    int i, neq;

    double residual;

    neq  = E->lmesh.NEQ;

    for(i=0;i<neq;i++)  {
        d0[i] = 0.0;
    }

    residual = sqrt(global_vdot(E,F,F));

    count = 0;

    cycles = E->control.v_steps_low;
    residual = conj_grad(E,d0,F,acc,&cycles);
    E->valid = (residual < acc)? 1:0;

    count++;

    E->monitor.momentum_residual = residual;
    E->control.total_iteration_cycles += count;
    E->control.total_v_solver_calls += 1;

    return;
}


/*------------------------------------------------------------------------*/

static void assert_assumptions(struct All_variables *E, int high_lev) {
    
    assert(!E->control.NMULTIGRID);
    
    assert(E->sphere.caps_per_proc == CAPS_PER_PROC);
    
    assert(E->mesh.nsd == NSD);
    assert(E->mesh.levmax == LEVEL);
    assert(high_lev == LEVEL);
    
    assert(E->parallel.nproc == 1);
    assert(E->parallel.TNUM_PASS[LEVEL][M] == 0);
    assert(E->parallel.Skip_neq[LEVEL][M] == 0);
}

__host__ int launch_solve_del2_u(
    struct All_variables *E,
    double **d0,
    double **F,
    double acc,
    int high_lev
    )
{
    struct Some_variables kE;
    struct octoterm *ot;
    int n_octoterms;
    
    assert_assumptions(E, high_lev);
    
    /* initialize 'Some_variables' with 'All_variables' */
    
    kE.num_zero_resid = E->num_zero_resid[LEVEL][M];
    kE.zero_resid = E->zero_resid[LEVEL][M];
        
    kE.lmesh.NEQ = E->lmesh.NEQ[LEVEL];
    kE.lmesh.NNO = E->lmesh.NNO[LEVEL];
    kE.lmesh.NEL = E->lmesh.NEL[LEVEL];
        
    kE.IEN   = E->IEN[LEVEL][M];
    kE.ID    = E->ID[LEVEL][M];
    kE.elt_k = E->elt_k[LEVEL][M];
        
    kE.Eqn_k1 = E->Eqn_k1[LEVEL][M];
    kE.Eqn_k2 = E->Eqn_k2[LEVEL][M];
    kE.Eqn_k3 = E->Eqn_k3[LEVEL][M];
    kE.Node_map = E->Node_map[LEVEL][M];
        
    kE.BI = E->BI[LEVEL][M];
        
    kE.control.NASSEMBLE = E->control.NASSEMBLE;
    kE.control.v_steps_low = E->control.v_steps_low;
    kE.control.total_iteration_cycles = E->control.total_iteration_cycles; /* in/out */
    kE.control.total_v_solver_calls = E->control.total_v_solver_calls; /* in/out */
    
    /* allocate temporary memory */
    kE.memoryDim = 0;
    kE.memoryDim += 5 * E->lmesh.NEQ[LEVEL]  + /* r0,r1,r2,z0,z1 */
                    3 * (1+E->lmesh.NEQ[LEVEL]) /* p1,p2,Ap */
                 ;
    kE.memory = (double *)malloc(kE.memoryDim*sizeof(double));
    
    kE.mm = (struct matrix_mult *)malloc(kE.lmesh.NEQ * sizeof(struct matrix_mult));
    n_octoterms = e_tally(&kE);
    ot = (struct octoterm *)malloc(n_octoterms * sizeof(struct octoterm));
    e_map(&kE, ot);
    
    /* zero outputs */
    kE.monitor.momentum_residual = 0.0;
    kE.valid = 0;
    
    /* solve */
    solve_del2_u(&kE, d0[M], F[M], acc);
    
    /* get outputs */
    E->control.total_iteration_cycles = kE.control.total_iteration_cycles;
    E->control.total_v_solver_calls = kE.control.total_v_solver_calls;
    E->monitor.momentum_residual = kE.monitor.momentum_residual;
    
    free(kE.memory);
    free(kE.mm);
    free(ot);
    
    return kE.valid;
}
