/* -*- C -*- */
/* vim:set ft=c: */

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
    
    higher_precision *Eqn_k1, *Eqn_k2, *Eqn_k3;
    int *Node_map;
    
    double *BI;
    
    double *temp;
    unsigned int *NODE;
};


/*------------------------------------------------------------------------*/
/* from BC_util.c */

__device__ void strip_bcs_from_residual(
    struct Some_variables *E,
    double *Res
    )
{
    int i;

    if (E->num_zero_resid)
        for (i=1;i<=E->num_zero_resid;i++)
            Res[E->zero_resid[i]] = 0.0;

    return;
}


/*------------------------------------------------------------------------*/
/* from Element_calculations.c */

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


    for (e=0;e<=neq;e++) {
        Au[e]=0.0;
    }

    u[neq] = 0.0;

    for (e=1;e<=nno;e++) {

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

        for (i=3;i<MAX_EQN;i++) {
            UU = u[C[i]];
            Au[eqn1] += B1[i]*UU;
            Au[eqn2] += B2[i]*UU;
            Au[eqn3] += B3[i]*UU;
        }
        for (i=0;i<MAX_EQN;i++) {
            Au[C[i]] += B1[i]*U1+B2[i]*U2+B3[i]*U3;
        }

    }     /* end for e */

    if (strip_bcs)
	strip_bcs_from_residual(E,Au);

    return;
}


/*------------------------------------------------------------------------*/
/* based on the function from General_matrix_functions.c */

__global__ void do_gauss_seidel(
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

    double UU;

    higher_precision *B1,*B2,*B3;


    const int neq=E->lmesh.NEQ;

    const double zeroo = 0.0;

    steps=*cycles;

    if (guess) {
        n_assemble_del2_u(E,d0,Ad,1);
    } else {
        for (i=0;i<neq;i++) {
            d0[i]=Ad[i]=zeroo;
        }
    }

    for (count = 0; count < steps; ++count) {
        for (j=0;j<=E->lmesh.NEQ;j++) {
            E->temp[j] = zeroo;
        }

        Ad[neq] = zeroo;

        for (i=1;i<=E->lmesh.NNO;i++) {
            if (E->NODE[i] & OFFSIDE) {

                eqn1=E->ID[i].doff[1];
                eqn2=E->ID[i].doff[2];
                eqn3=E->ID[i].doff[3];
                E->temp[eqn1] = (F[eqn1] - Ad[eqn1])*E->BI[eqn1];
                E->temp[eqn2] = (F[eqn2] - Ad[eqn2])*E->BI[eqn2];
                E->temp[eqn3] = (F[eqn3] - Ad[eqn3])*E->BI[eqn3];
            }
        }

        for (i=1;i<=E->lmesh.NNO;i++) {

            eqn1=E->ID[i].doff[1];
            eqn2=E->ID[i].doff[2];
            eqn3=E->ID[i].doff[3];
            C=E->Node_map+(i-1)*MAX_EQN;
            B1=E->Eqn_k1+(i-1)*MAX_EQN;
            B2=E->Eqn_k2+(i-1)*MAX_EQN;
            B3=E->Eqn_k3+(i-1)*MAX_EQN;

            /* Ad on boundaries differs after the following operation, but
               no communications are needed yet, because boundary Ad will
               not be used for the G-S iterations for interior nodes */

            for (j=3;j<MAX_EQN;j++)  {
                UU = E->temp[C[j]];
                Ad[eqn1] += B1[j]*UU;
                Ad[eqn2] += B2[j]*UU;
                Ad[eqn3] += B3[j]*UU;
            }

            if (!(E->NODE[i]&OFFSIDE))   {
                E->temp[eqn1] = (F[eqn1] - Ad[eqn1])*E->BI[eqn1];
                E->temp[eqn2] = (F[eqn2] - Ad[eqn2])*E->BI[eqn2];
                E->temp[eqn3] = (F[eqn3] - Ad[eqn3])*E->BI[eqn3];
            }

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
    
    kE.Eqn_k1 = E->Eqn_k1[level][M];
    kE.Eqn_k2 = E->Eqn_k2[level][M];
    kE.Eqn_k3 = E->Eqn_k3[level][M];
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
    assert(0);
}
