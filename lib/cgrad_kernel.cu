/* -*- C -*- */
/* vim:set ft=c: */

#include <math.h>
#include "global_defs.h"


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
    int ot_i; /* index of octoterm struct in 'ot' array */
    int zero_res; /* boolean */
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
    int n_octoterms;
    struct octoterm *ot;
    
    /* outputs */
    
    struct /*MONITOR*/ {
        double momentum_residual;
    } monitor;

    int valid;
};


/*------------------------------------------------------------------------*/
/* from BC_util.c */

void strip_bcs_from_residual(
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
        E->mm[i].zero_res = 0;
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
    
    /* strip_bcs_from_residual */
    for(i=1;i<=E->num_zero_resid;i++)
        E->mm[E->zero_resid[i]].zero_res = 1;

    /* return the total number of octoterms */
    return nTotal;
}

static void e_map(
    struct Some_variables *E
    )
{
    int  e,i,a,a1,a2,a3,o1,o2,o3,ii;
    struct octoterm *ot;

    const int nel=E->lmesh.NEL;
    const int neq=E->lmesh.NEQ;
    
    ot = E->ot;
    for(i=0;i<neq;i++) {
        E->mm[i].ot_i = ot - E->ot;
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
            
            ot = E->ot + E->mm[a1].ot_i + o1;
            ot->e = e;
            ot->a = a;
            ot->offset = 0;
            
            ot = E->ot + E->mm[a2].ot_i + o2;
            ot->e = e;
            ot->a = a;
            ot->offset = LOC_MAT_SIZE;
            
            ot = E->ot + E->mm[a3].ot_i + o3;
            ot->e = e;
            ot->a = a;
            ot->offset = LOC_MAT_SIZE + LOC_MAT_SIZE;
            
        }             /* end for loop a */

    }          /* end for e */
    
    return;
}

/* based on the function from Element_calculations.c */

__global__ void e_assemble_del2_u(
    struct Some_variables *E,
    double *u, double *Au,
    int strip_bcs
    )
{
    int  e,i,a,b,o,ii,nodeb,offset;
    struct octoterm *ot;
    
    /* ENODES*ENODES = 8*8 = 64 threads (2 warps) per block */
    __shared__ double sum[ENODES*ENODES];
    unsigned int tid = ENODES*threadIdx.y + threadIdx.x; /* 0 <= tid < 64 */
    
    do {
        i = blockIdx.x; /* 0 <= i < E->lmesh.NEQ */
        
        /***
         * This is block 'i'.
         */
        
        if (strip_bcs && E->mm[i].zero_res) {
            
            /* no-op: Au[i] is zero */
            sum[tid] = 0.0; /* but only sum[0] is used */
            
            /* XXX: How many blocks are wasted here? */
            
        } else {
            
            o = threadIdx.x; /* 0 <= o < ENODES */
            
            if (o < E->mm[i].n) {
                
                /****
                 * This is thread group 'o' in block 'i'.
                 * Each group of 8 threads handles one 'octoterm'.
                 */
                
                ot = E->ot + E->mm[i].ot_i + o;
                e      = ot->e;
                a      = ot->a;
                offset = ot->offset;
                
                do {
                    b = threadIdx.y + 1; /* 1 <= b <= ENODES */
                    
                    /****
                     * This is thread '(o, b)' in block 'i'.
                     * Each thread computes three terms.
                     */
                    
                    nodeb = E->IEN[e].node[b];
                    ii = (a*LOC_MAT_SIZE+b)*NSD-(NSD*LOC_MAT_SIZE+NSD);
                    
                    sum[tid] =
                        E->elt_k[e].k[ii+offset] *
                        u[E->ID[nodeb].doff[1]]
                        + E->elt_k[e].k[ii+offset+1] *
                        u[E->ID[nodeb].doff[2]]
                        + E->elt_k[e].k[ii+offset+2] *
                        u[E->ID[nodeb].doff[3]];
                    
                } while (0);
                
            } else {
                /* XXX: 8-n wasted threads per block go through here */
                sum[tid] = 0.0;
            }
            
            __syncthreads();
            
            /* Reduce the partial sums for this block.
               Based on reduce2() in the CUDA SDK. */
            for (unsigned int s = ENODES*ENODES / 2; s > 0; s >>= 1) 
            {
                if (tid < s) {
                    sum[tid] += sum[tid + s];
                }
                /* XXX: This is unnecessary, since only a single
                   "warp" is involved. */
                __syncthreads();
            }
        }
        
        /* XXX: ??? */
        __syncthreads();

        /* each block writes one element, Au[i], in global memory */
        if (tid == 0) {
            Au[i] = sum[0];
        }

    } while (0);

    return;
}


/*------------------------------------------------------------------------*/
/* from Global_operations.c */

double global_vdot(
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

static void construct_E(
    struct Some_variables **d_E,
    struct Some_variables *s_E, /* host's shadow copy of d_E */
    struct Some_variables *E
    )
{
    /* construct a copy of 'E' in device memory */
    
    int neq = E->lmesh.NEQ;
    int nno = E->lmesh.NNO;
    int nel = E->lmesh.NEL;
    
    memset(s_E, 0, sizeof(struct Some_variables));
    
    /* mm, ot */
    cudaMalloc((void**)&s_E->mm, neq * sizeof(struct matrix_mult));
    cudaMalloc((void**)&s_E->ot, E->n_octoterms * sizeof(struct octoterm));
    cudaMemcpy(s_E->mm, E->mm, neq * sizeof(struct matrix_mult), cudaMemcpyHostToDevice);
    cudaMemcpy(s_E->ot, E->ot, E->n_octoterms * sizeof(struct octoterm), cudaMemcpyHostToDevice);
    
    /* IEN -- cf. allocate_common_vars() */
    cudaMalloc((void**)&s_E->IEN, (nel+2)*sizeof(struct IEN));
    cudaMemcpy(s_E->IEN, E->IEN, (nel+2)*sizeof(struct IEN), cudaMemcpyHostToDevice);
    
    /* ID -- cf. allocate_common_vars()*/
    cudaMalloc((void **)&s_E->ID, (nno+1)*sizeof(struct ID));
    cudaMemcpy(s_E->ID, E->ID, (nno+1)*sizeof(struct ID), cudaMemcpyHostToDevice);
    
    /* elt_k -- cf. general_stokes_solver_setup() */
    cudaMalloc((void **)&s_E->elt_k, (nel+1)*sizeof(struct EK));
    cudaMemcpy(s_E->elt_k, E->elt_k, (nel+1)*sizeof(struct EK), cudaMemcpyHostToDevice);
    
    /* E */
    cudaMalloc((void**)d_E, sizeof(Some_variables));
    cudaMemcpy(*d_E, s_E, sizeof(Some_variables), cudaMemcpyHostToDevice);
    
    return;
}

static void destroy_E(
    struct Some_variables *d_E,
    struct Some_variables *s_E
    )
{
    cudaFree(s_E->mm);
    cudaFree(s_E->ot);
    cudaFree(s_E->IEN);
    cudaFree(s_E->ID);
    cudaFree(s_E->elt_k);
    cudaFree(d_E);
}


/*------------------------------------------------------------------------*/
/* from General_matix_functions.c */

double conj_grad(
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
    
    /* pointers to device memory */
    struct Some_variables *d_E = 0;
    double *d_p2 = 0, *d_Ap = 0;
    
    /* construct 'E' on the device */
    struct Some_variables s_E;
    construct_E(&d_E, &s_E, E);
    
    /* allocate memory on the device */
    cudaMalloc((void**)&d_p2, (1+neq)*sizeof(double));
    cudaMalloc((void**)&d_Ap, (1+neq)*sizeof(double));
    
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
        
        /********************************************/
        /* launch e_assemble_del2_u() on the device */
        
        /* copy input to the device */
        cudaMemcpy(d_p2, p2, (1+neq)*sizeof(double), cudaMemcpyHostToDevice);
        
        /* launch */
        dim3 dimBlock(ENODES, ENODES, 1);
        dim3 dimGrid(neq, 1, 1);
        e_assemble_del2_u<<< dimGrid, dimBlock >>>(d_E, d_p2, d_Ap, 1);
        
        /* wait for completion */
        cudaThreadSynchronize();
        
        /* copy output from device */
        cudaMemcpy(Ap, d_Ap, (1+neq)*sizeof(double), cudaMemcpyDeviceToHost);
        
        /********************************************/
        
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
    
    /* free device memory */
    cudaFree(d_p2);
    cudaFree(d_Ap);
    
    destroy_E(d_E, &s_E);

    *cycles=count;

    strip_bcs_from_residual(E,d0);
    
    return residual;
}


void do_solve_del2_u(
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
    assert(!E->control.NASSEMBLE);
    
    assert(E->sphere.caps_per_proc == CAPS_PER_PROC);
    
    assert(E->mesh.nsd == NSD);
    assert(E->mesh.levmax == LEVEL);
    assert(high_lev == LEVEL);
    
    assert(E->parallel.nproc == 1);
    assert(E->parallel.TNUM_PASS[LEVEL][M] == 0);
    assert(E->parallel.Skip_neq[LEVEL][M] == 0);
}

extern "C" int solve_del2_u(
    struct All_variables *E,
    double **d0,
    double **F,
    double acc,
    int high_lev
    )
{
    struct Some_variables kE;
    
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
    kE.n_octoterms = e_tally(&kE);
    kE.ot = (struct octoterm *)malloc(kE.n_octoterms * sizeof(struct octoterm));
    e_map(&kE);
    
    /* zero outputs */
    kE.monitor.momentum_residual = 0.0;
    kE.valid = 0;
    
    /* solve */
    do_solve_del2_u(&kE, d0[M], F[M], acc);
    
    /* get outputs */
    E->control.total_iteration_cycles = kE.control.total_iteration_cycles;
    E->control.total_v_solver_calls = kE.control.total_v_solver_calls;
    E->monitor.momentum_residual = kE.monitor.momentum_residual;
    
    free(kE.memory);
    free(kE.mm);
    free(kE.ot);
    
    return kE.valid;
}
