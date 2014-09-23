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

#include <stdio.h>
#include "global_defs.h"


void compute_horiz_avg(struct All_variables*);


/*
 * Given a radius (non-dimensional),
 * returns Vp/Vs/rho (in km/s and g/cm^3) of PREM.
 */
void get_prem(double r, double *vp, double *vs, double *rho)
{
#define NUM_PREM_LAYERS 11

/* some specfem flags */
#define SUPPRESS_CRUSTAL_MESH 0
#define ONE_CRUST 1

    /* radius of various layers */
    const double prem_radius[NUM_PREM_LAYERS] =
        {0.19164966253335425,  /* 0: ICB */
         0.54622508240464607,  /* 1: CMB */
         0.56976926699105324,  /* 2: top of D'' */
         0.87898289122586726,  /* 3: 771 */
         0.89483597551404803,  /* 4: 670 */
         0.90582326165437133,  /* 5: 600 */
         0.93721550776958096,  /* 6: 400 */
         0.96546852927326954,  /* 7: 220 */
         0.99617014597394449,  /* 8: Moho */
         0.99764558154135929,  /* 9: middle crust */
         1.00000000000000000}; /*10: top surface */

    const int j_cmb = 1;
    const int j_moho = 8;


    /* polynomial coefficients of PREM */
    const double prem_vs[NUM_PREM_LAYERS][4] =
        {{ 3.6678,   0.0000, -4.4475,  0.0000},
         { 0.0010,   0.0000,  0.0000,  0.0000},
         { 6.9254,   1.4672, -2.0834,  0.9783},
         {11.1671, -13.7818, 17.4575, -9.2777},
         {22.3459, -17.2473, -2.0834,  0.9783},
         { 9.9839,  -4.9324,  0.0000,  0.0000},
         {22.3512, -18.5856,  0.0000,  0.0000},
         { 8.9496,  -4.4597,  0.0000,  0.0000},
         { 2.1519,   2.3481,  0.0000,  0.0000},
         { 3.9000,   0.0000,  0.0000,  0.0000},
         { 3.2000,   0.0000,  0.0000,  0.0000}};

    const double prem_vp[NUM_PREM_LAYERS][4] =
        {{11.2622,   0.0000, -6.3640,  0.0000},
         {11.0487,  -4.0362,  4.8023,-13.5732},
         {15.3891,  -5.3181,  5.5242, -2.5514},
         {24.9520, -40.4673, 51.4832,-26.6419},
         {29.2766, -23.6027,  5.5242, -2.5514},
         {19.0957,  -9.8672,  0.0000,  0.0000},
         {39.7027, -32.6166,  0.0000,  0.0000},
         {20.3926, -12.2569,  0.0000,  0.0000},
         { 4.1875,   3.9382,  0.0000,  0.0000},
         { 6.8000,   0.0000,  0.0000,  0.0000},
         { 5.8000,   0.0000,  0.0000,  0.0000}};

    const double  prem_rho[NUM_PREM_LAYERS][4] =
        {{13.0885,   0.0000, -8.8381,  0.0000},
         {12.5815,  -1.2638, -3.6426, -5.5281},
         { 7.9565,  -6.4761,  5.5283, -3.0807},
         { 7.9565,  -6.4761,  5.5283, -3.0807},
         { 7.9565,  -6.4761,  5.5283, -3.0807},
         { 5.3197,  -1.4836,  0.0000,  0.0000},
         {11.2494,  -8.0298,  0.0000,  0.0000},
         { 7.1089,  -3.8045,  0.0000,  0.0000},
         { 2.6910,   0.6924,  0.0000,  0.0000},
         { 2.9000,   0.0000,  0.0000,  0.0000},
         { 2.6000,   0.0000,  0.0000,  0.0000}};


    int j;
    double r2, r3;

    /* make sure r is above CMB */
    r = (r < prem_radius[j_cmb]) ? prem_radius[j_cmb] : r;
    r2 = r * r;
    r3 = r2 * r;

    /* find layer */
    for (j = 0; j < NUM_PREM_LAYERS; ++j)
        if (r < prem_radius[j]) break;

    if (j < 0) j = 0;
    if (j >= NUM_PREM_LAYERS) j = NUM_PREM_LAYERS - 1;

    if(SUPPRESS_CRUSTAL_MESH && j > j_moho) {
        /* extend of Moho up to the surface instead of the crust */
        j = 8;
    }
    if(ONE_CRUST && j > j_moho) {
        /* replace mid-crust with upper crust */
        j = 10;
    }

    /* expand polynomials */
    *vp = prem_vp[j][0]
        + prem_vp[j][1] * r
        + prem_vp[j][2] * r2
        + prem_vp[j][3] * r3;
    *vs = prem_vs[j][0]
        + prem_vs[j][1] * r
        + prem_vs[j][2] * r2
        + prem_vs[j][3] * r3;
    *rho = prem_rho[j][0]
        + prem_rho[j][1] * r
        + prem_rho[j][2] * r2
        + prem_rho[j][3] * r3;

    /** debug **
    fprintf(stderr, "%e %d %f %f %f\n", r, j, *rho, *vp, *vs);
    */

#undef NUM_PREM_LAYERS
#undef SUPPRESS_CRUSTAL_MESH
#undef ONE_CRUST

}



static void modified_Trampert_Vacher_Vlaar_PEPI2001(struct All_variables *E,
                                                    double *rho, double *vp, double *vs)
{

    /* Table 2 in the paper, including quasi-harmonic and anelastic parts */
    const double dlnvpdt[3] = {-5.71e-5, 2.44e-8, -3.84e-12};
    const double dlnvsdt[3] = {-9.37e-5, 3.70e-8, -5.46e-12};
    const double dlnvpdc[3] = {1.72e-1, -0.98e-4, 1.44e-8};
    const double dlnvsdc[3] = {1.50e-1, -1.43e-4, 1.92e-8};

    const int m = 1;
    int i, j, nz;
    double *rhor, *vpr, *vsr, *depthkm;
    double d, d2, dT, dC, drho, dvp, dvs;

    /* compute horizontal average */
    if(!E->output.horiz_avg)
        compute_horiz_avg(E);


    /* reference model (PREM) */
    rhor = malloc((E->lmesh.noz+1) * sizeof(double));
    vpr = malloc((E->lmesh.noz+1) * sizeof(double));
    vsr = malloc((E->lmesh.noz+1) * sizeof(double));
    depthkm = malloc((E->lmesh.noz+1) * sizeof(double));

    for(nz=1; nz<=E->lmesh.noz; nz++) {
        get_prem(E->sx[3][nz], &vpr[nz], &vsr[nz], &rhor[nz]);
        depthkm[nz] = (1.0 - E->sx[3][nz]) * E->data.radius_km;
    }

    /* deviation from the reference */
    dC = 0;
    for(i=0; i<E->lmesh.nno; i++) {
        nz = (i % E->lmesh.noz) + 1;

        d = depthkm[nz];
        d2 = d * d;
        dT = (E->T[i+1] - E->Have.T[nz]) * E->data.ref_temperature;

        drho = -dT * E->refstate.thermal_expansivity[nz] * E->data.therm_exp;

        dvp = dT * (dlnvpdt[0] + dlnvpdt[1]*d + dlnvpdt[2]*d2);
        dvs = dT * (dlnvsdt[0] + dlnvsdt[1]*d + dlnvsdt[2]*d2);

        if(E->control.tracer && E->composition.on && E->composition.ichemical_buoyancy)
            for(j=0; j<E->composition.ncomp; j++) {
                dC = E->composition.comp_node[j][i+1] - E->Have.C[j][nz];

                drho += dC * E->composition.buoyancy_ratio[j]
                    * E->data.ref_temperature * E->data.therm_exp / E->refstate.rho[nz];

                dvp += dC * (dlnvpdc[0] + dlnvpdc[1]*d + dlnvpdc[2]*d2);
                dvs += dC * (dlnvsdc[0] + dlnvsdc[1]*d + dlnvsdc[2]*d2);
            }

        rho[i] = rhor[nz] * (1 + drho);
        vp[i] = vpr[nz] * (1 + dvp);
        vs[i] = vsr[nz] * (1 + dvs);

        /** debug **
        fprintf(stderr, "node=%d dT=%f K, dC=%f, %e %e %e\n",
                i, dT, dC, drho, dvp, dvs);
        */
    }


    free(rhor);
    free(vpr);
    free(vsr);
    free(depthkm);

}



void compute_seismic_model(struct All_variables *E,
                           double *rho, double *vp, double *vs)
{

    switch(E->control.mineral_physics_model) {

    case 0:
        /* reserved for Stixrude and Lithgow-Bertelloni, GJI, 2005 model */
        fprintf(stderr,"Invalid value: 'mineral_physics_model=%d'\n",
                E->control.mineral_physics_model);
        parallel_process_termination();

        break;

    case 1:
        /* reserved for Karato, GRL, 1993 model */
        fprintf(stderr,"Invalid value: 'mineral_physics_model=%d'\n",
                E->control.mineral_physics_model);
        parallel_process_termination();

        break;

    case 2:
        /* reserved for Stacy, PEPI, 1998 model */
        fprintf(stderr,"Invalid value: 'mineral_physics_model=%d'\n",
                E->control.mineral_physics_model);
        parallel_process_termination();

        break;

    case 3:
        /* Based on the paper:
         * Trampert, Vacher, and Vlaar, PEPI, 2001.
         *
         * Note that the paper has its own reference profile (which is not
         * shown in the paper), and is only valid between
         * 1000 km < depth < 2600 km.
         *
         * But here we use PREM as the reference model, and extend the model to the
         * whole mantle.
         */

        modified_Trampert_Vacher_Vlaar_PEPI2001(E, rho, vp, vs);
        break;

    case 100:
        /* user-defined mineral physics model goes here */
        fprintf(stderr,"Need user definition for mineral physics model: 'mineral_physics_model=%d'\n",
                E->control.mineral_physics_model);
        parallel_process_termination();
        break;

    default:
        /* unknown option */
        fprintf(stderr,"Invalid value: 'mineral_physics_model=%d'\n",
                E->control.mineral_physics_model);
        parallel_process_termination();
        break;
    }

}
