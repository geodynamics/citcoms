/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *<LicenseText>
 *=====================================================================
 *
 *                              CitcomS
 *                 ---------------------------------
 *
 *                              Authors:
 *           Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 *           Clint Conrad, Michael Gurnis, and Eun-seo Choi
 *          (c) California Institute of Technology 1994-2005
 *
 *        By downloading and/or installing this software you have
 *       agreed to the CitcomS.py-LICENSE bundled with this software.
 *             Free for non-commercial academic research ONLY.
 *      This program is distributed WITHOUT ANY WARRANTY whatsoever.
 *
 *=====================================================================
 *
 *  Copyright June 2005, by the California Institute of Technology.
 *  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
 * 
 *  Any commercial use must be negotiated with the Office of Technology
 *  Transfer at the California Institute of Technology. This software
 *  may be subject to U.S. export control laws and regulations. By
 *  accepting this software, the user agrees to comply with all
 *  applicable U.S. export laws and regulations, including the
 *  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
 *  the Export Administration Regulations, 15 C.F.R. 730-744. User has
 *  the responsibility to obtain export licenses, or other export
 *  authority as may be required before exporting such information to
 *  foreign countries or providing access to foreign nationals.  In no
 *  event shall the California Institute of Technology be liable to any
 *  party for direct, indirect, special, incidental or consequential
 *  damages, including lost profits, arising out of the use of this
 *  software and its documentation, even if the California Institute of
 *  Technology has been advised of the possibility of such damage.
 * 
 *  The California Institute of Technology specifically disclaims any
 *  warranties, including the implied warranties or merchantability and
 *  fitness for a particular purpose. The software and documentation
 *  provided hereunder is on an "as is" basis, and the California
 *  Institute of Technology has no obligations to provide maintenance,
 *  support, updates, enhancements or modifications.
 *
 *=====================================================================
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
