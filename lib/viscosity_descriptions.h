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
/* in this file define the contents of the VISC_OPT data structure
   which is used to store information used to create predefined
   viscosity fields, those determined from prior input, those
   related to temperature/pressure/stress/anything else. */


#define CITCOM_MAX_VISC_LAYER 40

struct VISC_OPT {
    void (* update_viscosity)();

    int update_allowed;		/* determines whether visc field can evolve */
    int EQUIVDD;			/* Whatever the structure, average in the end */
    int equivddopt;
    int proflocx;			/* use depth dependence from given x,y location */
    int proflocy;
    int SMOOTH;
    int smooth_cycles;


    char STRUCTURE[20];		/* which option to determine viscosity field, one of .... */
    int FROM_SYSTEM;
    int FROM_FILE;
    int FROM_SPECS;

				/* System ... */
    int RHEOL;			/* 1,2 */
    int rheol_layers;
    int num_mat;

    int ncmb;
    int nlm;
    int n410;
    int nlith;
    float zcmb;			/* old layer specs */
    float zlm;
    float z410;
    float zlith;
  float zbase_layer[CITCOM_MAX_VISC_LAYER]; /* new */

    int FREEZE;
    float freeze_thresh;
    float freeze_value;

    int channel;
    int wedge;

    float lv_min_radius;
    float lv_max_radius;
    float lv_channel_thickness;
    float lv_reduction;

    int MAX;
    float max_value;
    int MIN;
    float min_value;

    int SDEPV;
    float sdepv_misfit;
    int sdepv_normalize,sdepv_visited;
    float sdepv_expt[CITCOM_MAX_VISC_LAYER];
    float sdepv_trns[CITCOM_MAX_VISC_LAYER];


  int CDEPV;			/* compositional viscosity */
  float cdepv_ff[10];		/*  flavor factors */

  int PDEPV;			/* "plasticity" law parameters */
  float pdepv_a[CITCOM_MAX_VISC_LAYER], pdepv_b[CITCOM_MAX_VISC_LAYER], pdepv_y[CITCOM_MAX_VISC_LAYER],pdepv_offset;
  int pdepv_eff,pdepv_visited;

    int TDEPV;
    int TDEPV_AVE;
    float N0[CITCOM_MAX_VISC_LAYER];
    float E[CITCOM_MAX_VISC_LAYER],T0[CITCOM_MAX_VISC_LAYER];
    float T[CITCOM_MAX_VISC_LAYER],Z[CITCOM_MAX_VISC_LAYER];

    int weak_blobs;
    float weak_blobx[CITCOM_MAX_VISC_LAYER];
    float weak_bloby[CITCOM_MAX_VISC_LAYER];
    float weak_blobz[CITCOM_MAX_VISC_LAYER];
    float weak_blobwidth[CITCOM_MAX_VISC_LAYER];
    float weak_blobmag[CITCOM_MAX_VISC_LAYER];

    int weak_zones;
    float weak_zonex1[CITCOM_MAX_VISC_LAYER];
    float weak_zoney1[CITCOM_MAX_VISC_LAYER];
    float weak_zonez1[CITCOM_MAX_VISC_LAYER];
    float weak_zonex2[CITCOM_MAX_VISC_LAYER];
    float weak_zoney2[CITCOM_MAX_VISC_LAYER];
    float weak_zonez2[CITCOM_MAX_VISC_LAYER];

    float weak_zonewidth[CITCOM_MAX_VISC_LAYER];
    float weak_zonemag[CITCOM_MAX_VISC_LAYER];

    int guess;
    char old_file[100];
				/* Specification info */

				/* Prespecified viscosity parameters */
    char VISC_OPT[20];

  // superceded by num_mat
  //int layers;			/* number of layers with properties .... */


    int SLABLVZ;			/* slab structure imposed on top of 3 layer structure */
    int slvzd1,slvzd2,slvzd3;	        /* layer thicknesses (nodes) */
    int slvzD1,slvzD2;		        /* slab posn & length */
    float slvzn1,slvzn2,slvzn3,slvzN;   /* viscosities */

    int COSX;
    float cosx_epsilon;
    float cosx_k;
    int cosx_exp;

    int EXPX;
    float expx_epsilon;

  float ET_red,T_sol0;			/* for viscosity law 8 */


    /* MODULE BASED VISCOSITY VARIATIONS */

    int RESDEPV;
    float RESeta0[CITCOM_MAX_VISC_LAYER];

    int CHEMDEPV;
    float CH0[CITCOM_MAX_VISC_LAYER];
    float CHEMeta0[CITCOM_MAX_VISC_LAYER];

} viscosity;
