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


#define CITCOM_MAX_VISC_LAYER 40

struct VISC_OPT {

    int update_allowed;		/* determines whether visc field can evolve */
    int SMOOTH;
    int smooth_cycles;

  int allow_anisotropic_viscosity,anisotropic_viscosity_init;
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
  int anivisc_start_from_iso; /* start from isotropic solution? */
  int anisotropic_init;	/* 0: isotropic, 1: random, 2: from file */
  char anisotropic_init_dir[1000];
  int anivisc_layer;		/* layer to assign anisotropic viscosity to for mode 2 */
  double ani_vis2_factor;	/* for  mode 3, anisotropy scale factor*/
#endif

    char STRUCTURE[20];		/* which option to determine viscosity field, one of .... */
    int FROM_SYSTEM;
    int FROM_FILE;
    int FROM_SPECS;

    /* System ... */
    int RHEOL;			/* 1,2 */
    int num_mat;

    float zcmb;			/* old layer specs */
    float zlm;
    float z410;
    float zlith;
    float zbase_layer[CITCOM_MAX_VISC_LAYER]; /* new */


    /* low viscosity channel and wedge stuff */
    int channel;
    int wedge;

    float lv_min_radius;
    float lv_max_radius;
    float lv_channel_thickness;
    float lv_reduction;


    /* viscosity cut-off */
    int MAX;
    float max_value;
    int MIN;
    float min_value;


    /* non-Newtonian stress dependence */
    int SDEPV;
    float sdepv_misfit;
    int sdepv_normalize, sdepv_visited,sdepv_rheol,sdepv_start_from_newtonian;

    float *sdepv_expt,*sdepv_trns;
  
    int visc_adjust_N0;

    /* compositional viscosity */
    int CDEPV;
    float cdepv_ff[10];		/*  flavor factors */


    /* "plasticity" law parameters */
    int PDEPV;
    float *pdepv_a, *pdepv_b, *pdepv_y;
    float pdepv_offset;
    int pdepv_eff, pdepv_visited;
    int psrw;


    /* temperature dependence */
    int TDEPV;
    float *N0, *E, *T, *Z;

    float ET_red, T_sol0;			/* for viscosity law 8 */

    int layer_control;
    char layer_file[255];

} viscosity;
