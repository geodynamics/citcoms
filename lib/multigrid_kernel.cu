/* -*- C -*- */
/* vim:set ft=c: */

#if __CUDA_ARCH__ < 130
/* for double-precision floating-point */
#error This code requires compute capability 1.3 or higher; try giving "-arch sm_13".
#endif


#include "global_defs.h"
#include "element_definitions.h"
#include <assert.h>
#include <stdio.h>


enum {
    CAPS_PER_PROC = 1,
    M = 1, /* cap # */
    NSD = 3, /* Spatial extent: 3d */
    MAX_EQN = NSD*14,
};


struct Some_variables {
    int num_zero_resid;
    int *zero_resid;
    
    struct /*MESH_DATA*/ {
        int NEQ;
        int NNO;
    } lmesh;
    
    struct ID *ID;
    
    higher_precision *Eqn_k[NSD+1];
    int *Node_map;
    
    double *BI;
    
    double *temp;
    unsigned int *NODE;
    
    int2 **term;
};


/*------------------------------------------------------------------------*/
/* from Element_calculations.c */

__global__ void n_assemble_del2_u(
    struct Some_variables *E,
    double *u, double *Au,
    int strip_bcs
    )
{
    int n = blockIdx.x + 1; /* 1 <= n <= E->lmesh.NNO */
    int doff = blockIdx.y + 1; /* 1 <= doff < NSD */ 
    unsigned int tid = threadIdx.x; /* 0 <= tid < MAX_EQN */
    
    /* Each block writes one element of Au in global memory: Au[eqn]. */
    int eqn = E->ID[n].doff[doff]; /* XXX: Compute this value? */
    
    if (strip_bcs) {
        /* See get_bcs_id_for_residual(). */
        unsigned int flags = E->NODE[n];
        unsigned int vb = 0x1 << doff; /* VBX, VBY, or VBZ */
        if (flags & vb) {
            /* no-op: Au[eqn] is zero */
            if (tid == 0) {
                Au[eqn] = 0.0;
            }
            /* XXX: Hundreds of blocks exit here (E->num_zero_resid).
               Does it matter? */
            return;
        }
    }
    
    /* The partial sum computed by this thread. */
    double acc;
    
    /* Part I: The terms here are easily derived from the block and
       thread indices. */
    {
        int e = n; /* 1 <= e <= E->lmesh.NNO */
        int i = (int)tid; /* 0 <= i < MAX_EQN */
        
        if (i < 3) {
            acc = 0.0;
        } else {
            int *C = E->Node_map + (e-1)*MAX_EQN;
            higher_precision *B = E->Eqn_k[doff]+(e-1)*MAX_EQN;
            double UU = u[C[i]];
            acc = B[i]*UU;
        }
    }
    
    /* Part II: These terms are more complicated. */
    {
        int2 pair = E->term[eqn][tid];
        int e = pair.x; /* 1 <= e <= E->lmesh.NNO */
        int i = pair.y; /* 0 <= i < MAX_EQN */
        
        if (i != -1) {
            /* XXX: Compute these values? */
            int eqn1 = E->ID[e].doff[1];
            int eqn2 = E->ID[e].doff[2];
            int eqn3 = E->ID[e].doff[3];
            
            double U1 = u[eqn1];
            double U2 = u[eqn2];
            double U3 = u[eqn3];
            
            higher_precision *B1, *B2, *B3;
            B1 = E->Eqn_k[1]+(e-1)*MAX_EQN;
            B2 = E->Eqn_k[2]+(e-1)*MAX_EQN;
            B3 = E->Eqn_k[3]+(e-1)*MAX_EQN;
            
            acc += B1[i]*U1 +
                   B2[i]*U2 +
                   B3[i]*U3;
        } else {
            /* XXX: A considerable number of threads idle here. */
        }
    }
    
    /* Reduce the partial sums for this block.
       Based on reduce2() in the CUDA SDK. */
    __shared__ double sum[MAX_EQN];
    sum[tid] = acc;
    __syncthreads();
    for (unsigned int s = MAX_EQN/2; s > 0; s >>= 1) {
        if (tid < s) {
            sum[tid] += sum[tid + s];
        }
        /* XXX: not always necessary */
        __syncthreads();
    }
    
    /* Each block writes one element of Au in global memory. */
    if (tid == 0) {
        Au[eqn] = sum[0];
    }
    
    return;
}


/*------------------------------------------------------------------------*/
/* These are based on the function from General_matrix_functions.c. */

__global__ void gauss_seidel_0(
    struct Some_variables *E,
    double *d0,
    double *Ad
    )
{
    const double zeroo = 0.0;
    int i;
    
    i = blockIdx.x; /* 0 <= i < E->lmesh.NEQ */
    d0[i] = Ad[i] = zeroo;
}

__global__ void gauss_seidel_1(
    struct Some_variables *E,
    double *F, double *Ad
    )
{
    const double zeroo = 0.0;
    const int neq = E->lmesh.NEQ;
    
    int i, doff, eqn;
    
    i = blockIdx.x + 1; /* 1 <= i <= E->lmesh.NNO */
    doff = blockIdx.y + 1; /* 1 <= doff < NSD */ 
    eqn = E->ID[i].doff[doff];
    
    if (E->NODE[i] & OFFSIDE) {
        E->temp[eqn] = (F[eqn] - Ad[eqn])*E->BI[eqn];
    } else {
        E->temp[eqn] = zeroo;
    }
    
    if (i == 1 && doff == 1) {
        E->temp[neq] = zeroo;
        Ad[neq] = zeroo;
    }
}

__global__ void gauss_seidel_2(
    struct Some_variables *E,
    double *F, double *Ad
    )
{
    int i, doff, eqn;
    
    i = blockIdx.x + 1; /* 1 <= i <= E->lmesh.NNO */
    doff = blockIdx.y + 1; /* 1 <= doff < NSD */ 
    eqn = E->ID[i].doff[doff];
    
    int *C;
    higher_precision *B;
    double UU, Ad_eqn;
    int j;
    
    C = E->Node_map+(i-1)*MAX_EQN;
    B = E->Eqn_k[doff]+(i-1)*MAX_EQN;
    
    /* load from global memory */
    Ad_eqn = Ad[eqn];
    
    /* Ad on boundaries differs after the following operation, but
       no communications are needed yet, because boundary Ad will
       not be used for the G-S iterations for interior nodes */
    
    for (j=3;j<MAX_EQN;j++)  {
        UU = E->temp[C[j]];
        Ad_eqn += B[j]*UU;
    }
    
    /* store to global memory */
    Ad[eqn] = Ad_eqn;
    
    if (!(E->NODE[i] & OFFSIDE))   {
        E->temp[eqn] = (F[eqn] - Ad_eqn)*E->BI[eqn];
    }

}

void do_gauss_seidel(
    struct Some_variables *E,
    double *d0,
    double *F, double *Ad,
    double acc,
    int *cycles,
    int guess
    )
{

    int count,i,j,steps;
    int *C;
    int eqn1,eqn2,eqn3;

    higher_precision *B1,*B2,*B3;

    steps=*cycles;

    dim3 neqBlock(1, 1, 1);
    dim3 neqGrid(E->lmesh.NEQ, 1, 1);
    
    dim3 nnoBlock(1, 1, 1);
    dim3 nnoGrid(E->lmesh.NNO, NSD, 1);
    
    /* XXX: allocate & init device memory */
    struct Some_variables *d_E = 0;
    double *d_d0 = 0, *d_F = 0, *d_Ad = 0;
    
    if (guess) {
        /* XXX */
        d_Ad[E->lmesh.NEQ] = 0.0; /* Au -- unnecessary? */
        d_d0[E->lmesh.NEQ] = 0.0; /* u */
        
        dim3 block(MAX_EQN, 1, 1);
        dim3 grid(E->lmesh.NNO, NSD, 1);
        n_assemble_del2_u<<< grid, block >>>(d_E, d_d0, d_Ad, 1);
    } else {
        gauss_seidel_0<<< neqGrid, neqBlock >>>(d_E, d_d0, d_Ad);
    }
    
    for (count = 0; count < steps; ++count) {
        
        gauss_seidel_1<<< nnoGrid, nnoBlock >>>(d_E, d_F, d_Ad);
        gauss_seidel_2<<< nnoGrid, nnoBlock >>>(d_E, d_F, d_Ad);
        
        
        /* XXX: How to parallelize this? */
        for (i=1;i<=E->lmesh.NNO;i++) {

            /* Ad on boundaries differs after the following operation */
            for (j=0;j<MAX_EQN;j++) {
                Ad[C[j]]  += B1[j]*E->temp[eqn1]
                             +  B2[j]*E->temp[eqn2]
                             +  B3[j]*E->temp[eqn3];
            }

            d0[eqn1] += E->temp[eqn1];
            d0[eqn2] += E->temp[eqn2];
            d0[eqn3] += E->temp[eqn3];
        }
    }
    
    /* wait for completion */
    cudaThreadSynchronize();
    
    *cycles=count;
    return;
}


/*------------------------------------------------------------------------*/

static void assert_assumptions(struct All_variables *E, int level) {
    
    assert(E->control.NMULTIGRID);
    
    assert(E->sphere.caps_per_proc == CAPS_PER_PROC);
    
    assert(E->mesh.nsd == NSD);
    
    assert(E->parallel.nproc == 1);
}

static void tally_n_assemble_del2_u(
    struct Some_variables *E //,
    //double *u, double *Au,
    //int strip_bcs
    )
{
    int e,i;
    int eqn1,eqn2,eqn3;
    
#if 0
    double UU,U1,U2,U3;
#endif

    int *C;
#if 0
    higher_precision *B1,*B2,*B3;
#endif

    const int neq=E->lmesh.NEQ;
    const int nno=E->lmesh.NNO;
    
    /*
     * Au = E->Eqn_k? * u
     *  where E->Eqn_k? is the sparse stiffness matrix
     */
    
    int maxAcc, total;
    int *tally;
    int **threadMap, **threadTally;
    int2 **terms;
    int f;
    
    tally = (int *)malloc((neq+1) * sizeof(int));
    threadMap = (int **)malloc((neq+1)* sizeof(int*));
    threadTally = (int **)malloc((neq+1)* sizeof(int*));
    terms = (int2 **)malloc((neq+1)* sizeof(int2 *));
    
    for(e=0;e<=neq;e++) {
        //Au[e]=0.0;
        tally[e] = 0;
        threadMap[e] = (int *)malloc((neq+1) * sizeof(int));
        threadTally[e] = (int *)malloc((neq+1) * sizeof(int));
        terms[e] = (int2 *)malloc((MAX_EQN+1) * sizeof(int2));
        for(f=0;f<=neq;f++) {
            threadMap[e][f] = -1;
            threadTally[e][f] = 0;
        }
        for (f = 0; f < MAX_EQN; ++f) {
            terms[e][f].x = -1;
            terms[e][f].y = -1;
        }
        terms[e][MAX_EQN].x = 0;
        terms[e][MAX_EQN].y = 0;
    }

#if 0
    u[neq] = 0.0;
#endif

    for(e=1;e<=nno;e++)     {

        eqn1=E->ID[e].doff[1];
        eqn2=E->ID[e].doff[2];
        eqn3=E->ID[e].doff[3];
        
        /* could compute, but 'Node_map' is more complicated */
        assert(eqn1 == 3*(e-1));
        assert(eqn2 == eqn1+1);
        assert(eqn3 == eqn1+2);
        
        /* could put maps in constant memory */
        
        /*
         * Key observation: after parallelizing on 'e' (either one):
         *
         *   ID[e].doff[1,2,3]
         *   C
         *
         * are fixed for each thread.  => Not worth obsessing over?
         */
        
        /*
         * Put Au[eqnX] into shared memory; it is accessed almost MAX_EQN=42 times.
         *
         * "Au[e]=0.0" should be unnecessary -- single write at end of fn
         * from 'AuX' local var.. so actually, Au[eqnX] sits in register
         * 
         * But what about "Au[C[i]]"???????????
         */
        
        /*
         * neq vs. nno
         *
         * neq == 3*nno
         * warp=32; only 2 threads wasted
         * better: 3 warps 32*3
         *
         * use y for "dimension index"? (block size 32x3; nno % 32)
         */
        
#if 0
        U1 = u[eqn1];
        U2 = u[eqn2];
        U3 = u[eqn3];
#endif

        C=E->Node_map + (e-1)*MAX_EQN;
#if 0
        B1=E->Eqn_k[1]+(e-1)*MAX_EQN;
        B2=E->Eqn_k[2]+(e-1)*MAX_EQN;
        B3=E->Eqn_k[3]+(e-1)*MAX_EQN;
#endif

        for(i=3;i<MAX_EQN;i++)  {
#if 0
            UU = u[C[i]];
            Au[eqn1] += B1[i]*UU;
            Au[eqn2] += B2[i]*UU;
            Au[eqn3] += B3[i]*UU;
#endif
            ++tally[eqn1];
            ++tally[eqn2];
            ++tally[eqn3];
            for(f=0;f<=neq;f++) {
                if (threadMap[eqn1][f] == e) {
                    ++threadTally[eqn1][f];
                    break;
                }
                if (threadMap[eqn1][f] == -1) {
                    threadMap[eqn1][f] = e;
                    ++threadTally[eqn1][f];
                    break;
                }
            }
            for(f=0;f<=neq;f++) {
                if (threadMap[eqn2][f] == e) {
                    ++threadTally[eqn2][f];
                    break;
                }
                if (threadMap[eqn2][f] == -1) {
                    threadMap[eqn2][f] = e;
                    ++threadTally[eqn2][f];
                    break;
                }
            }
            for(f=0;f<=neq;f++) {
                if (threadMap[eqn3][f] == e) {
                    ++threadTally[eqn3][f];
                    break;
                }
                if (threadMap[eqn3][f] == -1) {
                    threadMap[eqn3][f] = e;
                    ++threadTally[eqn3][f];
                    break;
                }
            }
        }
        for(i=0;i<MAX_EQN;i++) {
#if 0
            Au[C[i]] += B1[i]*U1+B2[i]*U2+B3[i]*U3;
#endif
            ++tally[C[i]];
            for(f=0;f<=neq;f++) {
                if (threadMap[C[i]][f] == e) {
                    ++threadTally[C[i]][f];
                    break;
                }
                if (threadMap[C[i]][f] == -1) {
                    threadMap[C[i]][f] = e;
                    ++threadTally[C[i]][f];
                    break;
                }
            }
            ++terms[C[i]][MAX_EQN].y;
            for (f = 0; f < MAX_EQN; ++f) {
                if (terms[C[i]][f].y == -1) {
                    terms[C[i]][f].x = e;
                    terms[C[i]][f].y = i;
                    break;
                }
            }
            assert(C[i] == neq || f < MAX_EQN);
        }

    }     /* end for e */
    
    maxAcc = 0;
    total = 0;
    for(e=0;e<=neq;e++) {
        int myTally;
        fprintf(stderr, "Au[%d]: %d times", e, tally[e]);
        if (e < neq)
            maxAcc = max(maxAcc, tally[e]);
        total += tally[e];
        myTally = 0;
        for(f=0;f<=neq;f++) {
            if (threadMap[e][f] == -1)
                break;
            fprintf(stderr, " %d(%d)", threadMap[e][f], threadTally[e][f]);
            myTally += threadTally[e][f];
        }
        fprintf(stderr, " (%d times)\n", myTally);
    }
    //fprintf(stderr, "Au[%d] == %f\n", e - 1, Au[e]);
    fprintf(stderr, "max accesses %d\n", maxAcc);
    fprintf(stderr, "total accesses %d\n", total);
    
    fprintf(stderr, "\nterms:\n");
    for(e=0;e<=neq;e++) {
        fprintf(stderr, "Au[%d]: %d terms %s", e, terms[e][MAX_EQN].y,
                terms[e][MAX_EQN].y > MAX_EQN ? "XXXTO" : "");
        for (f = 0; f < MAX_EQN; ++f) {
            if (terms[e][f].y == -1)
                break;
            fprintf(stderr, " %d(%d)", terms[e][f].y, terms[e][f].x);
        }
        fprintf(stderr, "\n");
    }
    
#if 0
    if (strip_bcs)
        strip_bcs_from_residual(E,Au);
#endif

    return;
}

extern "C" void gauss_seidel(
    struct All_variables *E,
    double **d0,
    double **F, double **Ad,
    double acc,
    int *cycles,
    int level,
    int guess
    )
{
    struct Some_variables kE;
    
    assert_assumptions(E, level);
    
    /* initialize 'Some_variables' with 'All_variables' */
    
    kE.num_zero_resid = E->num_zero_resid[level][M];
    kE.zero_resid = E->zero_resid[level][M];
    
    kE.lmesh.NEQ = E->lmesh.NEQ[level];
    kE.lmesh.NNO = E->lmesh.NNO[level];
    
    kE.ID    = E->ID[level][M];
    
    kE.Eqn_k[0] = 0;
    kE.Eqn_k[1] = E->Eqn_k1[level][M];
    kE.Eqn_k[2] = E->Eqn_k2[level][M];
    kE.Eqn_k[3] = E->Eqn_k3[level][M];
    kE.Node_map = E->Node_map[level][M];
    
    kE.BI = E->BI[level][M];
    
    kE.temp = E->temp[M];
    
    kE.NODE = E->NODE[level][M];
    
                                       /* XXX */
    do {
        int i, doff, print;
        for (i=1;i<=kE.lmesh.NNO;i++) {
            print = (i < 10 || i > kE.lmesh.NNO - 10);
            if (print)
                fprintf(stderr, "%04d:", i);
            for (doff = 1; doff <= 3; ++doff) {
                assert(kE.ID[i].doff[doff] == /*NSD*/ 3 * (i - 1) + doff - 1);
                if (print)
                    fprintf(stderr, " %d", kE.ID[i].doff[doff]);
            }
            if (print)
                fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n0 - NEQ %d\n", kE.lmesh.NEQ);
    } while (0);
    tally_n_assemble_del2_u(&kE);
    do {
        int i;
        fprintf(stderr, "E->num_zero_resid == %d\n", kE.num_zero_resid);
        for (i=1;i<=kE.num_zero_resid;i++)
            fprintf(stderr, "    Au[%d] = 0.0\n", kE.zero_resid[i]);
    } while (0);
    assert(0);
}
