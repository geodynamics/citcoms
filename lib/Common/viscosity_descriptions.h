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
    float zcmb;
    float zlm;
    float z410;
    float zlith;

    int FREEZE;
    float freeze_thresh;
    float freeze_value;

    int MAX;
    float max_value;
    int MIN;
    float min_value;

    int SDEPV;
    float sdepv_misfit;
    int sdepv_normalize;
    float sdepv_expt[40];
    float sdepv_trns[40];

    int TDEPV;
    int TDEPV_AVE;
    float N0[40];
    float E[40],T0[40];
    float T[40],Z[40];

    int weak_blobs;
    float weak_blobx[40];
    float weak_bloby[40];
    float weak_blobz[40];
    float weak_blobwidth[40];
    float weak_blobmag[40];
   
    int weak_zones;
    float weak_zonex1[40];
    float weak_zoney1[40];
    float weak_zonez1[40];
    float weak_zonex2[40];
    float weak_zoney2[40];
    float weak_zonez2[40];
  
    float weak_zonewidth[40];
    float weak_zonemag[40];
  
    int guess;
    char old_file[100];
				/* Specification info */
  
				/* Prespecified viscosity parameters */
    char VISC_OPT[20];

    int layers;			/* number of layers with properties .... */
    float layer_depth[40];
    float layer_visc[40];

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
 
    /* MODULE BASED VISCOSITY VARIATIONS */

    int RESDEPV;
    float RESeta0[40];

    int CHEMDEPV;
    float CH0[40];
    float CHEMeta0[40];
  
} viscosity;
