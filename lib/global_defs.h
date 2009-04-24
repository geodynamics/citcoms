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
#if !defined(CitcomS_global_defs_h)
#define CitcomS_global_defs_h

/*
This file contains the definitions of variables which are passed as arguments
to functions across the whole filespace of CITCOM.
#include this file everywhere !
*/
#ifdef USE_GGRD
#include "hc.h"
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"



#ifdef USE_HDF5
#include "hdf5.h"
#endif

#ifdef __cplusplus

extern "C" {

#else




/* Macros */
#define max(A,B) (((A) > (B)) ? (A) : (B))
#define min(A,B) (((A) < (B)) ? (A) : (B))
#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

#endif


#define LIDN 0x1
#define VBX 0x2
#define VBZ 0x4
#define VBY 0x8
#define TBX 0x10
#define TBZ 0x20
#define TBY 0x40
#define TZEDGE 0x80
#define TXEDGE 0x100
#define TYEDGE 0x200
#define VXEDGE 0x400
#define VZEDGE 0x800
#define VYEDGE 0x1000
#define INTX 0x2000
#define INTZ 0x4000
#define INTY 0x8000
#define SBX 0x10000
#define SBZ 0x20000
#define SBY 0x40000
#define FBX 0x80000
#define FBZ 0x100000
#define FBY 0x200000

#define OFFSIDE 0x400000
#define PMARGINS 0x800000
#define SSLAB 0x400000

#define SKIP 0x1000000
#define SKIPS 0x2000000
#define SKIPID 0x4000000
#define ZEROID 0x8000000


#ifndef COMPRESS_BINARY
#define COMPRESS_BINARY "/usr/bin/compress"
#endif

#define MAX_LEVELS 12   /* max. number of multigrid levels */
#define NCS      14   /* max. number of sphere caps */

/* type of elt_del and elt_c arrays */
/* double precision doesn't help,
 * probably due to the coordinate transformation c33matrix */
#if 1
    typedef float higher_precision;
#else
    typedef double higher_precision;
#endif


/* Common structures */

struct Bdry {
  int nel;
  int *element[NCS];
  int *normal[NCS][4];
  double *det[NCS][7][5];
};


struct SBC {
  /* stress (traction) boundary conditions */
  int *node[NCS];
  double *SB[NCS][7][4];
};


struct Shape_function_dA  {
  double vpt[8+1];
  double ppt[1+1]; };

struct Shape_function1_dA  {
  double vpt[6*4];
  double ppt[6*1]; };

struct Shape_function_side_dA  {
  double vpt[4];
  double ppt[1]; };

struct Shape_function1 	{
    double vpt[4*4];  /* node & gauss pt */
    double ppt[4*1];  };

struct Shape_function 	{
    double vpt[8*8];  /* node & gauss pt */
    double ppt[8*1];  };

struct Shape_function_dx 	{
    double vpt[3*8*8]; /* dirn & node & gauss pt */
    double ppt[3*8*1];  };

struct Shape_function1_dx 	{
    double vpt[2*4*4]; /* dirn & node & gauss pt */
    double ppt[2*4*1];  };

struct CC 	{
    double vpt[3*3*8*8];
    double ppt[3*3*8*1]; }; /* dirn & node & gauss pt */

struct CCX 	{
    double vpt[3*3*2*8*8];
    double ppt[3*3*2*8*1]; }; /* dirn & node & gauss pt */

struct EG {
    higher_precision g[24][1]; };

struct EC {
    higher_precision c[24][1]; };

struct EK {
    double k[24*24]; };

struct COORD {
    float centre[4];
    float size[4];
    float area;   } ;

struct SUBEL {
    int sub[9];   };

struct ID  {
    int doff[4];	}; /* can  be 1 or 2 or 3 */

struct IEN {
    int node[9];	};

struct FNODE {
    float node[9];	};

struct SIEN {
    int node[5];	};

struct BOUND  {
    int bound[8];	};

struct PASS  {
    int pass[27];	};

struct Parallel {
    MPI_Comm world;
    MPI_Comm horizontal_comm;
    MPI_Comm vertical_comm;

    int me;
    int nproc;
    int nprocx;
    int nprocz;
    int nprocy;
    int nprocxy;
    int nproczy;
    int nprocxz;

    int total_surf_proc;
    int ****loc2proc_map;


    int redundant[MAX_LEVELS];
    int idb;
    int me_loc[4];
    int num_b;
    int Skip_neq[MAX_LEVELS][NCS];
    int *Skip_id[MAX_LEVELS][NCS];

    int TNUM_PASS[MAX_LEVELS][NCS];
    struct BOUND *NODE[MAX_LEVELS][NCS];
    struct BOUND NUM_NNO[MAX_LEVELS][NCS];
    struct BOUND NUM_PASS[MAX_LEVELS][NCS];
    struct PASS NUM_NEQ[MAX_LEVELS][NCS];
    struct PASS NUM_NODE[MAX_LEVELS][NCS];
    struct PASS PROCESSOR[MAX_LEVELS][NCS];
    struct PASS *EXCHANGE_ID[MAX_LEVELS][NCS];
    struct PASS *EXCHANGE_NODE[MAX_LEVELS][NCS];

    int TNUM_PASSz[MAX_LEVELS];
    struct BOUND NUM_PASSz[MAX_LEVELS];
    struct PASS PROCESSORz[MAX_LEVELS];
    struct PASS NUM_NEQz[MAX_LEVELS];
    struct PASS NUM_NODEz[MAX_LEVELS];

    int sTNUM_PASS[MAX_LEVELS][NCS];
    struct PASS NUM_sNODE[MAX_LEVELS][NCS];
    struct PASS sPROCESSOR[MAX_LEVELS][NCS];
    struct PASS *EXCHANGE_sNODE[MAX_LEVELS][NCS];
    };

struct CAP    {
    double theta[5];
    double fi   [5];
    float *TB[4];
    float *VB[4];
    float *V[4];
    float *Vprev[4];
    };

struct SPHERE   {
  int caps;
  int caps_per_proc;
  int capid[NCS];
  int max_connections;
  int *hindex[100];
  int hindice;
  float *harm_geoid[2];
  float *harm_geoid_from_bncy[2];
  float *harm_geoid_from_bncy_botm[2];
  float *harm_geoid_from_tpgt[2];
  float *harm_geoid_from_tpgb[2];

  float *harm_tpgt[2];
  float *harm_tpgb[2];

  double **tablesplm[NCS];
  double **tablescosf[NCS];
  double **tablessinf[NCS];

  double area[NCS];
  double angle[NCS][5];
  double *area1[MAX_LEVELS][NCS];
  double *angle1[MAX_LEVELS][NCS][5];

  double *R[MAX_LEVELS];
  double ro,ri;
  struct CAP cap[NCS];

};


struct MESH_DATA {/* general information concerning the fe mesh */
    int nsd;        /* Spatial extent 1,2,3d*/
    int dof;        /* degrees of freedom per node */
    int levmax;
    int levmin;
    int gridmax;
    int gridmin;
    int levels;
    int mgunitx;
    int mgunitz;
    int mgunity;
    int NEQ[MAX_LEVELS];
    int NNO[MAX_LEVELS];
    int NNOV[MAX_LEVELS];
    int NPNO[MAX_LEVELS];
    int NEL[MAX_LEVELS];
    int NOX[MAX_LEVELS];
    int NOZ[MAX_LEVELS];
    int NOY[MAX_LEVELS];
    int ELX[MAX_LEVELS];
    int ELZ[MAX_LEVELS];
    int ELY[MAX_LEVELS];
    int SNEL[MAX_LEVELS];
    int neq;
    int nno;
    int nnov;
    int npno;
    int nel;
    int snel;
    int elx;
    int elz;
    int ely;
    int nox;
    int noz;
    int noy;
    int exs;
    int ezs;
    int eys;
    int nxs;
    int nzs;
    int nys;
    int EXS[MAX_LEVELS];
    int EYS[MAX_LEVELS];
    int EZS[MAX_LEVELS];
    int NXS[MAX_LEVELS];
    int NYS[MAX_LEVELS];
    int NZS[MAX_LEVELS];
    int nsf; /* nodes for surface observables */
    int toptbc;
    int bottbc;
    int topvbc;
    int botvbc;
    int sidevbc;


    int periodic_x;
    int periodic_y;
    float layer[4];			/* dimensionless dimensions */
    double volume;
    int matrix_size[MAX_LEVELS];

} ;

struct HAVE {    /* horizontal averages */
    float *T;
    float *V[4];
    float **C;
};

struct SLICE {    /* horizontally sliced data, including topography */
    float *tpg[NCS];
    float *tpgb[NCS];
    float *shflux[NCS];
    float *bhflux[NCS];
    float *divg[NCS];
    float *vort[NCS];
    float *freesurf[NCS];
  };


struct MONITOR {
    int solution_cycles;
    int solution_cycles_init;

    int stop_topo_loop;
    int topo_loop;

    double momentum_residual;
    double incompressibility;
    double fdotf;
    double vdotv;
    double pdotp;

    double cpu_time_at_start;
    double cpu_time_at_last_cycle;
    float  elapsed_time;

    float T_interior;
    float T_maxvaried;
    float T_interior_max_for_exit;
};

struct CONTROL {
    int PID;

    char data_prefix[50];
    char data_prefix_old[50];

    char data_dir[150];
    char data_dir_old[150];

    char data_file[200];
    char old_P_file[200];

    char PROBLEM_TYPE[20]; /* one of ... */
    int CONVECTION;
    int stokes;
    int restart;
    int post_p;

    char GEOMETRY[20]; /* one of ... */
    int CART2D;
    int CART2pt5D;
    int CART3D;
    int AXI;

    char SOLVER_TYPE[20]; /* one of ... */
    int CONJ_GRAD;
    int NMULTIGRID;

    int pseudo_free_surf;

    int tracer;
    int tracer_enriched;

    double theta_min, theta_max, fi_min, fi_max;
    float start_age;
    int reset_startage;
    int zero_elapsed_time;

    float Ra_670,clapeyron670,transT670,inv_width670;
    float Ra_410,clapeyron410,transT410,inv_width410;
    float Ra_cmb,clapeyroncmb,transTcmb,inv_widthcmb;

    int augmented_Lagr;
    double augmented;
    int NASSEMBLE;

    float sob_tolerance;

    /* Rayleigh # */
    float Atemp;

    /* Dissipation # */
    float disptn_number;

    /* inverse of Gruneisen parameter */
    float inv_gruneisen;

    /* surface temperature */
    float surface_temp;

    /* adiabatic temperature extrapolated to the surface */
    /* float adiabaticT0; */

    /**/
    int compress_iter_maxstep;

  int self_gravitation;		/* self gravitation */
  int use_cbf_topo;		/* use consistent dynamic topo method? */

    char uzawa[20];

    float inputdiff;
    float VBXtopval;
    float VBXbotval;
    float VBYtopval;
    float VBYbotval;

    int coor;
    float coor_refine[4];
  float rrlayer[20];
  int nrlayer[20],rlayers;

    char coor_file[100];

  //int remove_hor_buoy_avg;

    float mantle_temp;

    int lith_age;
    int lith_age_time;
    int lith_age_old_cycles;
    float lith_age_depth;

  int precise_strain_rate; /* use proper computation for strain-rates in whole domain, not just poles */

    int temperature_bound_adj;
    float depth_bound_adj;
    float width_bound_adj;

    float TBCtopval;
    float TBCbotval;

    float Q0;
    float Q0ER;

    int precondition;
    int keep_going;
    int v_steps_low;
    int v_steps_high;
    int v_steps_upper;
    int p_iterations;
    int mg_cycle;
    int max_mg_cycles;
    int down_heavy;
    int up_heavy;
    int verbose;

    int remove_rigid_rotation;
    int remove_angular_momentum;

    int side_sbcs;
    int vbcs_file;
    int tbcs_file;
    int mat_control;
    int mineral_physics_model;
#ifdef USE_GGRD
  struct ggrd_master ggrd;
  float *surface_rayleigh;
  int ggrd_allow_mixed_vbcs;
  float ggrd_vtop_omega[4];
#endif
    double accuracy;
  int only_check_vel_convergence;
    char velocity_boundary_file[1000];
    char temperature_boundary_file[1000];
    char mat_file[1000];
    char lith_age_file[1000];

    int total_iteration_cycles;
    int total_v_solver_calls;

    int checkpoint_frequency;
    int record_every;
    int record_all_until;

    int print_convergence;
    int sdepv_print_convergence;

};


struct REF_STATE {
    int choice;
    char filename[200];
    double *rho;
    double *thermal_expansivity;
    double *heat_capacity;
    /*double *thermal_conductivity;*/
    double *gravity;
    /*double *Tadi;*/
};


struct DATA {
    float  layer_km;
    float  radius_km;
    float   grav_acc;
    float   therm_exp;
    float   Cp;
    float  therm_diff;
#ifdef ALLOW_ELLIPTICAL
  double  ellipticity, ra,rc,rotm,j2,ge,efac; /* for ellipticity tests: f, normalized a and c axes, 
						 rotational fraction m, J2, and norm gravity at the equator */
  int use_ellipse,use_rotation_g;
#endif
    float  therm_cond;
    float   density;
    float  res_density;
    float  res_density_X;
    float   melt_density;
    float   density_above;
    float   density_below;
    float   gas_const;
    float   surf_heat_flux;
    float  ref_viscosity;
    float   melt_viscosity;
    float   permeability;
    float   grav_const;
    float   youngs_mod;
    float   Te;
    float   ref_temperature;
    float   Tsurf;
    float   T_sol0;
    float   delta_S;
    float   dTsol_dz;
    float   dTsol_dF;
    float   dT_dz;
    float   scalet;
    float   scalev;
    float   timedir;
};

  /* for gzdir  */
struct gzd_struc{
  int vtk_io,vtk_base_init,vtk_base_save,
    vtk_ocount;
  float *vtk_base;
  FILE *vtk_fp;

  int rnr;			/* remove net rotation? */

};

struct Output {
    char format[20];  /* ascii or hdf5 */
    char optional[1000]; /* comma-delimited list of objects to output */

    int llmax;  /* max degree of spherical harmonics output */

    /* size of collective buffer used by MPI-IO */
    int cb_block_size;
    int cb_buffer_size;

    /* size of data sieve buffer used by HDF5 */
    int sieve_buf_size;

    /* memory alignment used by HDF5 */
    int alignment;
    int alignment_threshold;

    /* cache for chunked dataset used by HDF5 */
    int cache_mdc_nelmts;
    int cache_rdcc_nelmts;
    int cache_rdcc_nbytes;

    int connectivity; /* whether to output connectivity */
    int stress;       /* whether to output stress */
    int pressure;     /* whether to output pressure */
    int surf;         /* whether to output surface data */
    int botm;         /* whether to output bottom data */
    int geoid;        /* whether to output geoid/topo spherial harmonics */
    int horiz_avg;    /* whether to output horizontal averaged profile */
    int seismic;      /* whether to output seismic velocity model */
    int coord_bin;    /* whether to output coordinates in binary format */
    int tracer;       /* whether to output tracer coordinate */
    int comp_el;      /* whether to output composition at elements */
    int comp_nd;      /* whether to output composition at nodes */
    int heating;      /* whether to output heating terms at elements */


  /* flags used by GZDIR */
  struct gzd_struc gzdir;


  int write_q_files;
  FILE *fpqt,*fpqb;		/* additional heat flux output */
};


struct COMPOSITION {

    int ichemical_buoyancy;
    int icompositional_rheology;

    /* if any of the above flags is true, turn this flag on */
    int on;

    int ibuoy_type;
    int ncomp;
    double *buoyancy_ratio;

    double **comp_el[13];
    double **comp_node[13];

    double *initial_bulk_composition;
    double *bulk_composition;
    double *error_fraction;

    double compositional_rheology_prefactor;
};


struct CITCOM_GNOMONIC {
    /* gnomonic projected coordinate */
    double u;
    double v;
};


#ifdef USE_HDF5
#include "hdf5_related.h"
#endif
#include "tracer_defs.h"

struct All_variables {

#include "solver.h"
#include "convection_variables.h"
#include "viscosity_descriptions.h"
#include "advection.h"

    FILE *fp;
    FILE *fptime;
    FILE *fp_out;

#ifdef USE_HDF5
    struct HDF5_INFO hdf5;
#endif
    struct HAVE Have;
    struct MESH_DATA mesh;
    struct MESH_DATA lmesh;
    struct CONTROL control;
    struct MONITOR monitor;
    struct DATA data;
    struct SLICE slice;
    struct Parallel parallel;
    struct SPHERE sphere;
    struct Bdry boundary;
    struct SBC sbc;
    struct Output output;

    struct TRACE trace;

    /* for chemical convection & composition rheology */
    struct COMPOSITION composition;

    struct CITCOM_GNOMONIC *gnomonic;
    double gnomonic_reference_phi;

    struct COORD *eco[NCS];
    struct IEN *ien[NCS];  /* global */
    struct SIEN *sien[NCS];
    struct ID *id[NCS];
    struct COORD *ECO[MAX_LEVELS][NCS];
    struct IEN *IEN[MAX_LEVELS][NCS]; /* global at each level */
    struct FNODE *TWW[MAX_LEVELS][NCS];	/* for nodal averages */
    struct ID *ID[MAX_LEVELS][NCS];
    struct SUBEL *EL[MAX_LEVELS][NCS];
    struct EG *elt_del[MAX_LEVELS][NCS];
    struct EC *elt_c[MAX_LEVELS][NCS];
    struct EK *elt_k[MAX_LEVELS][NCS];
    struct CC *cc[NCS];
    struct CCX *ccx[NCS];
    struct CC *CC[MAX_LEVELS][NCS];
    struct CCX *CCX[MAX_LEVELS][NCS];

    struct CC element_Cc;
    struct CCX element_Ccx;

    struct REF_STATE refstate;


    higher_precision *Eqn_k1[MAX_LEVELS][NCS],*Eqn_k2[MAX_LEVELS][NCS],*Eqn_k3[MAX_LEVELS][NCS];
    int *Node_map [MAX_LEVELS][NCS];

    double *BI[MAX_LEVELS][NCS],*BPI[MAX_LEVELS][NCS];

    double *rho;
    double *heating_adi[NCS];
    double *heating_visc[NCS];
    double *heating_latent[NCS];

    double *P[NCS],*F[NCS],*U[NCS];
    double *T[NCS],*Tdot[NCS],*buoyancy[NCS];
    double *u1[NCS];
    double *temp[NCS],*temp1[NCS];
    double *Mass[NCS], *MASS[MAX_LEVELS][NCS];
    double *TMass[NCS], *NMass[NCS];
    double *SX[MAX_LEVELS][NCS][4],*X[MAX_LEVELS][NCS][4];
    double *sx[NCS][4],*x[NCS][4];
    double *surf_det[NCS][5];
    double *SinCos[MAX_LEVELS][NCS][4];

    float *NP[NCS];
  //float *stress[NCS];
    float *gstress[NCS];
    float *Fas670[NCS],*Fas410[NCS],*Fas670_b[NCS],*Fas410_b[NCS];
    float *Fascmb[NCS],*Fascmb_b[NCS];

    float *Vi[NCS],*EVi[NCS];
    float *VI[MAX_LEVELS][NCS],*EVI[MAX_LEVELS][NCS];

    int num_zero_resid[MAX_LEVELS][NCS];
    int *zero_resid[MAX_LEVELS][NCS];
    int *surf_element[NCS],*surf_node[NCS];
    int *mat[NCS];
    float *VIP[NCS];
    unsigned int *NODE[MAX_LEVELS][NCS];
    unsigned int *node[NCS];

    float *age_t;

    struct Shape_function_dx *GNX[MAX_LEVELS][NCS];
    struct Shape_function_dA *GDA[MAX_LEVELS][NCS];
    struct Shape_function_dx *gNX[NCS];
    struct Shape_function_dA *gDA[NCS];

    struct Shape_function1 M; /* master-element shape funtions */
    struct Shape_function1_dx Mx;
    struct Shape_function N;
    struct Shape_function NM;
    struct Shape_function_dx Nx;
    struct Shape_function1 L; /* master-element shape funtions */
    struct Shape_function1_dx Lx;
    struct Shape_function_dx NMx;

    void (* build_forcing_term)(void*);
    void (* iterative_solver)(void*);
    void (* next_buoyancy_field)(void*);
    void (* next_buoyancy_field_init)(void*);
    void (* obtain_gravity)(void*);
    void (* problem_settings)(void*);
    void (* problem_derived_values)(void*);
    void (* problem_allocate_vars)(void*);
    void (* problem_boundary_conds)(void*);
    void (* problem_update_node_positions)(void*);
    void (* problem_initial_fields)(void*);
    void (* problem_tracer_setup)(void*);
    void (* problem_tracer_output)(void*, int);
    void (* problem_update_bcs)(void*);
    void (* special_process_new_velocity)(void*);
    void (* special_process_new_buoyancy)(void*);
    void (* solve_stokes_problem)(void*);
    void (* solver_allocate_vars)(void*);
    void (* transform)(void*);

    float (* node_space_function[3])(void*);

    /* function pointer for choosing between various output routines */
    void (* problem_output)(struct All_variables *, int);

  /* the following function pointers are for exchanger */
  void (* exchange_node_d)(struct All_variables *, double**, int);
  void (* exchange_node_f)(struct All_variables *, float**, int);
  void (* temperatures_conform_bcs)(void*);

};

#ifdef __cplusplus
}
#endif




#endif
