/* -*- C -*- */
/* vim:set ft=c: */

#include "global_defs.h"
#include "element_definitions.h"


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

__device__ void n_assemble_del2_u(
    struct All_variables *E,
    double **u, double **Au,
    int level,
    int strip_bcs
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

    (E->solver.exchange_id_d)(E, Au, level);

    if (strip_bcs)
	strip_bcs_from_residual(E,Au,level);

    return;
}


/*------------------------------------------------------------------------*/
/* based on the function from General_matrix_functions.c */

__global__ void do_gauss_seidel(
    struct All_variables *E,
    double **d0,
    double **F, double **Ad,
    double acc,
    int *cycles,
    int level,
    int guess
    )
{

    int count,i,j,m,steps;
    int *C;
    int eqn1,eqn2,eqn3;

    double UU;

    higher_precision *B1,*B2,*B3;


    const int dims=E->mesh.nsd;
    const int neq=E->lmesh.NEQ[level];
    const int max_eqn=14*dims;

    const double zeroo = 0.0;

    steps=*cycles;

    if(guess) {
        n_assemble_del2_u(E,d0,Ad,level,1);
    }
    else
        for (m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=0;i<neq;i++) {
                d0[m][i]=Ad[m][i]=zeroo;
            }

    count = 0;


    while (count < steps) {
        for (m=1;m<=E->sphere.caps_per_proc;m++)
            for(j=0;j<=E->lmesh.NEQ[level];j++)
                E->temp[m][j] = zeroo;

        for (m=1;m<=E->sphere.caps_per_proc;m++)
            Ad[m][neq] = zeroo;

        for (m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=E->lmesh.NNO[level];i++)
                if(E->NODE[level][m][i] & OFFSIDE)   {

                    eqn1=E->ID[level][m][i].doff[1];
                    eqn2=E->ID[level][m][i].doff[2];
                    eqn3=E->ID[level][m][i].doff[3];
                    E->temp[m][eqn1] = (F[m][eqn1] - Ad[m][eqn1])*E->BI[level][m][eqn1];
                    E->temp[m][eqn2] = (F[m][eqn2] - Ad[m][eqn2])*E->BI[level][m][eqn2];
                    E->temp[m][eqn3] = (F[m][eqn3] - Ad[m][eqn3])*E->BI[level][m][eqn3];
                    E->temp1[m][eqn1] = Ad[m][eqn1];
                    E->temp1[m][eqn2] = Ad[m][eqn2];
                    E->temp1[m][eqn3] = Ad[m][eqn3];
                }

        for (m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=E->lmesh.NNO[level];i++)     {

                eqn1=E->ID[level][m][i].doff[1];
                eqn2=E->ID[level][m][i].doff[2];
                eqn3=E->ID[level][m][i].doff[3];
                C=E->Node_map[level][m]+(i-1)*max_eqn;
                B1=E->Eqn_k1[level][m]+(i-1)*max_eqn;
                B2=E->Eqn_k2[level][m]+(i-1)*max_eqn;
                B3=E->Eqn_k3[level][m]+(i-1)*max_eqn;

                /* Ad on boundaries differs after the following operation, but
                   no communications are needed yet, because boundary Ad will
                   not be used for the G-S iterations for interior nodes */

                for(j=3;j<max_eqn;j++)  {
                    UU = E->temp[m][C[j]];
                    Ad[m][eqn1] += B1[j]*UU;
                    Ad[m][eqn2] += B2[j]*UU;
                    Ad[m][eqn3] += B3[j]*UU;
                }

                if (!(E->NODE[level][m][i]&OFFSIDE))   {
                    E->temp[m][eqn1] = (F[m][eqn1] - Ad[m][eqn1])*E->BI[level][m][eqn1];
                    E->temp[m][eqn2] = (F[m][eqn2] - Ad[m][eqn2])*E->BI[level][m][eqn2];
                    E->temp[m][eqn3] = (F[m][eqn3] - Ad[m][eqn3])*E->BI[level][m][eqn3];
                }

                /* Ad on boundaries differs after the following operation */
                for(j=0;j<max_eqn;j++)
		    Ad[m][C[j]]  += B1[j]*E->temp[m][eqn1]
                                    +  B2[j]*E->temp[m][eqn2]
                                    +  B3[j]*E->temp[m][eqn3];

                d0[m][eqn1] += E->temp[m][eqn1];
                d0[m][eqn2] += E->temp[m][eqn2];
                d0[m][eqn3] += E->temp[m][eqn3];
  	    }

        for (m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=E->lmesh.NNO[level];i++)
                if(E->NODE[level][m][i] & OFFSIDE)   {
                    eqn1=E->ID[level][m][i].doff[1];
                    eqn2=E->ID[level][m][i].doff[2];
                    eqn3=E->ID[level][m][i].doff[3];
                    Ad[m][eqn1] -= E->temp1[m][eqn1];
                    Ad[m][eqn2] -= E->temp1[m][eqn2];
                    Ad[m][eqn3] -= E->temp1[m][eqn3];
                }

        (E->solver.exchange_id_d)(E, Ad, level);

        for (m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=E->lmesh.NNO[level];i++)
                if(E->NODE[level][m][i] & OFFSIDE)   {
                    eqn1=E->ID[level][m][i].doff[1];
                    eqn2=E->ID[level][m][i].doff[2];
                    eqn3=E->ID[level][m][i].doff[3];
                    Ad[m][eqn1] += E->temp1[m][eqn1];
                    Ad[m][eqn2] += E->temp1[m][eqn2];
                    Ad[m][eqn3] += E->temp1[m][eqn3];
                }


	count++;

        /*     for (m=1;m<=E->sphere.caps_per_proc;m++)
               for(i=0;i<neq;i++)          {
               F[m][i] -= Ad[m][i];
               Ad[m][i] = 0.0;
               }
        */
    }

    *cycles=count;
    return;

}


/*------------------------------------------------------------------------*/

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
    /* XXX */
}
