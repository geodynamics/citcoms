 	/* This file contains the definitions of variables which are passed as arguments */
	/* to functions across the whole filespace of CITCOM. #include this file everywhere !*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "mpi.h"

/*#include "/home/limbo1/louis/Software/include/dmalloc.h" */

#if defined(__osf__) 
void *Malloc1();
#endif

#define Malloc0(a) Malloc1((a),__FILE__,__LINE__)


/* #define Malloc0 malloc */

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

#define LIDE 1

#ifndef COMPRESS_BINARY
#define COMPRESS_BINARY "/usr/bin/compress"
#endif

#define MAX_LEVELS 12
#define MAX_F    10
#define MAX_S    30
#define NCS      14
#define MAXP 20

/* Macros */

#define max(A,B) (((A) > (B)) ? (A) : (B))
#define min(A,B) (((A) < (B)) ? (A) : (B))
#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

typedef float higher_precision;  /* matrix coeffs etc */
typedef double higher_precision1; /* intermediate calculations for finding above coeffs */


/* Common structures */

struct Rect {
    int numb;
    char overlay[40];
    float x1[40];
    float x2[40];
    float z1[40];
    float z2[40];
    float y1[40];
    float y2[40];
    float halfw[40];
    float mag[40];
} ;
 

struct Circ {
    int numb;
    char overlay[40];
    float x[40];
    float z[40];
    float y[40];
    float rad[40];
    float mag[40];
    float halfw[40];
};


struct Harm {
    int numb;
    int harms;
    char overlay[40];
    float off[40];
    float x1[40];
    float x2[40];
    float z1[40];
    float z2[40];
    float y1[40];
    float y2[40];
    float kx[20][40];
    float kz[20][40];
    float ky[20][40];
    float ka[20][40];
    float phx[20][40];
    float phz[20][40];
    float phy[20][40];
};

struct Erfc {
 int numb;


};

struct RectBc {
    int numb;
    char norm[40];
    float intercept[40];
    float x1[40];
    float x2[40];
    float z1[40];
    float z2[40];
    float halfw[40];
    float mag[40];
} ;
 

struct CircBc {
    int numb;
    char norm[40];
    float intercept[40];
    float x[40];
    float z[40];
    float rad[40];
    float mag[40];
    float halfw[40];
};


struct PolyBc {
    int numb;
    int order;
    char norm[40];
    float intercept[40];
    float x1[40];
    float x2[40];
    float z1[40];
    float z2[40];
    float ax[20][40];
    float az[20][40];
};


struct HarmBc {
    int numb;
    int harms;
    char norm[40];
    float off[40];
    float intercept[40];
    float x1[40];
    float x2[40];
    float z1[40];
    float z2[40];
    float kx[20][40];
    float kz[20][40];
    float ka[20][40];
    float phx[20][40];
    float phz[20][40];
 };


struct Shape_function_dA  {
  double vpt[8];
  double spt[4];
  double ppt[1]; };

struct Shape_function1_dA  {
  double vpt[6*4];
  double ppt[6*1]; };

struct Shape_function1 	{ 
    double vpt[4*4];  /* node & gauss pt */
    double ppt[4*1];  };

struct Shape_function 	{ 
    double vpt[8*8];  /* node & gauss pt */
    double spt[8*4];  /* node & gauss pt */
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

struct EK2 { 
    double k[8*8]; };

struct EK { 
    double k[24*24]; };

struct MEK { 
    double nint[9]; };
 
struct NK {
    higher_precision *k;
    int *map;
};

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
struct LM  { 
    struct { int doff[4]; } node[9]; } ;

struct NEI {
    int *nels;
    int *lnode;
    int *element; };

struct Crust  { float   width;
                float   thickness;
                float   *T;
		int     total_node;
		int     *node;};

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

    int nprocl;
    int nprocxl;
    int nproczl;
    int nprocyl;
    int nprocxyl;
    int nproczyl;
    int nprocxzl;
    int me_locl[4];

    int nproc_sph[3];
    int me_sph;
    int me_loc_sph[3];

    int redundant[MAX_LEVELS];
    int idb;
    int me_loc[4];
    int num_b;
    int Skip_neq[MAX_LEVELS][NCS];
    int *Skip_id[MAX_LEVELS][NCS];

    int mst[NCS][NCS];
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
    int connection[7];
    };

struct SPHERE   {
  int caps;
  int caps_per_proc;
  int local_capid[NCS];
  int capid[NCS];
  int pid_surf[NCS];
  int max_connections;
  int nox;
  int noy;
  int noz;
  int nsf; 
  int nno; 
  int elx;
  int ely;
  int elz;
  int snel;
  int llmax;
  int *int_cap;
  int *int_ele;
  int *hindex[100];
  int hindice;
  float *harm_tpgt[2];
  float *harm_tpgb[2];
  float *harm_slab[2];
  float *harm_velp[2];
  float *harm_velt[2];
  float *harm_divg[2];
  float *harm_vort[2];
  float *harm_visc[2];
  float *harm_geoid[2];
  double *sx[4];
  double *con;
  double *det[5];
  double *tableplm[181];
  double *tablecosf[361];
  double *tablesinf[361];
  double tablesint[181];

  double *tableplm_n[182];
  double *tablecosf_n[362];
  double *tablesinf_n[362];
  struct SIEN *sien;
  
  double area[NCS];
  double angle[NCS][5];
  double *area1[MAX_LEVELS][NCS];
  double *angle1[MAX_LEVELS][NCS][5];

  double dircos[4][4];
  double *R[MAX_LEVELS],*R_redundant;
  double ro,ri;
  struct CAP cap[NCS];

  float *radius;
  int slab_layers;
  int output_llmax;


  int lnox;
  int lnoy;
  int lnoz;
  int lnsf; 
  int lnno; 
  int lelx;
  int lely;
  int lelz;
  int lsnel;
  int lexs;
  int leys;
  };

struct MODEL  {
  int plates;
  int plate_node[MAXP];
  int *slab[MAXP];
  int dplate_node;
  int d410_node;
  int d670_node;
  float *plate_coord[MAXP][3];
  float *slab_age[MAXP];
  float plate_tfmax[MAXP][3];
  float plate_tfmin[MAXP][3];
  float plate_thickness;
  float plate_vel;
  float pmargin_width;
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
    int NLNO[MAX_LEVELS];
    int NPNO[MAX_LEVELS];
    int NEL[MAX_LEVELS];
    int NOX[MAX_LEVELS];
    int NOZ[MAX_LEVELS];
    int NOY[MAX_LEVELS];
    int NMX[MAX_LEVELS];
    int ELX[MAX_LEVELS];
    int ELZ[MAX_LEVELS];
    int ELY[MAX_LEVELS];
    int LNDS[MAX_LEVELS];
    int LELS[MAX_LEVELS];
    int SNEL[MAX_LEVELS];
    int neqd;
    int neq;
    int nno;
    int nnov;
    int nlno;
    int npno;
    int nel;
    int snel;
    int elx;
    int elz;
    int ely;
    int nnx[4]; /* general form of ... */
    int gnox;
    int gelx;
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
    int nmx;
    int nsf; /* nodes for surface observables */
    int toptbc;
    int bottbc;
    int topvbc;
    int botvbc;
    int sidevbc;

    char topvbc_file[100];
    char botvbc_file[100];
    char sidevbc_file[100];
    char gridfile[4][100];


    int periodic_x;
    int periodic_y;
    float layer[4];			/* dimensionless dimensions */
    float lidz;
    float bl1width[4],bl2width[4],bl1mag[4],bl2mag[4];
    float hwidth[4],magnitude[4],offset[4],width[4]; /* grid compression information */ 
    int fnodal_malloc_size;
    int dnodal_malloc_size;
    int feqn_malloc_size;
    int deqn_malloc_size;
    int bandwidth;
    int null_source;
    int null_sink;
    int matrix_size[MAX_LEVELS];

} ;

struct HAVE {    /* horizontal averages */
    float *T;
    float *Vi;
    float *Rho;
    float *f;
    float *F;
    float *vrms;
    float *V[4];
};

struct SLICE {    /* horizontally sliced data, including topography */
    float *tpg[NCS];
    float *tpgb[NCS];
    float *shflux[NCS];
    float *bhflux[NCS];
    float *divg[NCS];
    float *vort[NCS];
  };

struct BAVE {
    float T;
    float Vi;
    double V[4]; };


struct TOTAL {
    float melt_prod;  };

struct MONITOR {
    char node_output[100][6];  /* recording the format of the output data */
    char sobs_output[100][6];  /* recording the format of the output data */
    int node_output_cols;
    int sobs_output_cols;

    int solution_cycles;
    int solution_cycles_init;
  

    float  time_scale;
    float  length_scale;
    float  viscosity_scale;
    float  geoscale;
    float  tpgscale;
    float  grvscale;
  
    float  delta_v_last_soln;
    float  elapsed_time;
    float  elapsed_time_vsoln;
    float  elapsed_time_vsoln1;
    float  reference_stress;
    float  incompressibility;
    float  vdotv;
    float  nond_av_heat_fl;  
    float  nond_av_adv_hfl;  
    float  cpu_time_elapsed;
    float  cpu_time_on_vp_it;
    float  cpu_time_on_forces;
    float  cpu_time_on_mg_maps;
    float  tpgkmag;
    float  grvkmag;
   
    float  Nusselt;
    float  Vmax;
    float  Vsrms;
    float  Vrms;
    float  Vrms_surface;
    float  Vrms_base;
    float  F_surface;
    float  F_base;
    float  Frat_surface;
    float  Frat_base;
    float  T_interior;
    float  T_maxvaried;
    float  Sigma_max;
    float  Sigma_interior;
    float  Vi_average;
   
};

struct CONTROL {
    int PID;

    char output_written_external_command[500];   /* a unix command to run when output files have been created */

    int ORTHO,ORTHOZ;   /* indicates levels of mesh symmetry */
    char B_is_good[MAX_LEVELS];  /* general information controlling program flow */
    char Ahat_is_good[MAX_LEVELS];  /* general information controlling program flow */
    char old_P_file[100];
    char data_file[100];
    char post_topo_file[100];
    char slabgeoid_file[100];

    char which_data_files[1000];
    char which_horiz_averages[1000];
    char which_running_data[1000];
    char which_observable_data[1000];
  
    char PROBLEM_TYPE[20]; /* one of ... */
    int KERNEL;
    int CONVECTION;
    int stokes;
    int restart;
    int post_p;
    int post_topo;
    int SLAB;	
    char GEOMETRY[20]; /* one of ... */
    int CART2D;
    int CART2pt5D;
    int CART3D;
    int AXI;	 
    char SOLVER_TYPE[20]; /* one of ... */
    int DIRECT;
    int CONJ_GRAD;
    int NMULTIGRID;
    int EMULTIGRID;
    int DIRECTII;
    char NODE_SPACING[20]; /* turns into ... */
    int GRID_TYPE;
    int COMPRESS;
    int AVS;
    int CONMAN;
    int read_density;
    int read_slab;
    int read_slabgeoid;
    int tracer;
 

    float theta_min, theta_max, fi_min, fi_max;
    float start_age;
    int reset_startage;
    int zero_elapsed_time;

    float Ra_670,clapeyron670,transT670,width670;
    float Ra_410,clapeyron410,transT410,width410;
    float Ra_cmb,clapeyroncmb,transTcmb,widthcmb;

    int dfact;
    double penalty;
    int augmented_Lagr;
    double augmented;
    int NASSEMBLE;
    int crust;

    float tole_comp;
  
    float sob_tolerance;
 
    float Atemp; 
    float inputdiff; 
    float VBXtopval;
    float VBXbotval;
    float VBYtopval;
    float VBYbotval;

    int coor;
    char coor_file[100];

    char tracer_file[100];

    int lith_age;
    int lith_age_time;
    float lith_age_depth;
    float lith_age_mantle_temp;
    int temperature_bound_adj;
    float depth_bound_adj;
    float width_bound_adj;
    float TBCtopval;
    float TBCbotval;

    float Q0;
    float jrelax;
    
    int precondition;
    int vprecondition;
    int keep_going;
    int v_steps_low;
    int v_steps_high;
    int v_steps_upper;
    int max_vel_iterations;
    int p_iterations;
    int max_same_visc;
    float max_res_red_each_p_mg;
    float sub_stepping_factor;
    int mg_cycle;
    int true_vcycle;
    int down_heavy;
    int up_heavy;
    int depth_dominated;
    int eqn_viscosity;
    int eqn_zigzag;
    int verbose;
  /* input info */
  int VERBOSE;
  int DESCRIBE;
  int BEGINNER;

    int vbcs_file;
    int mat_control;
    double accuracy;
    double vaccuracy; 
    char velocity_boundary_file[1000];
    char mat_file[1000];
    char lith_age_file[1000];

    int total_iteration_cycles;
    int total_v_solver_calls;
    
    int record_every;
    int record_all_until;

    int print_convergence;
    int sdepv_print_convergence;

     /* modules */
    int MELTING_MODULE;
    int CHEMISTRY_MODULE;

};

struct DATA {  
    float  layer_km;
    float  radius_km;
    float   grav_acc;
    float   therm_exp;
    float   Cp;
    float  therm_diff;
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
    float  surf_temp;
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
	      
struct All_variables {     
#include "convection_variables.h"
#include "viscosity_descriptions.h"
#include "temperature_descriptions.h"
#include "advection.h"
#include "tracer_defs.h"

    FILE *fp;
    FILE *fptime;
    FILE *fp_out;
    struct HAVE Have;
    struct BAVE Bulkave;
    struct TOTAL Total;
    struct MESH_DATA mesh;
    struct MESH_DATA lmesh;
    struct CONTROL control;
    struct MONITOR monitor;
    struct DATA data;
    struct SLICE slice;
    struct Parallel parallel;
    struct SPHERE sphere;
    struct MODEL model;
    struct Tracer Tracer;

    int filed[20];
   
    struct COORD *eco[NCS];
    struct IEN *ien[NCS];  /* global */
    struct SIEN *sien[NCS];
    struct ID *id[NCS];
    struct COORD *ECO[MAX_LEVELS][NCS];
    struct IEN *IEN_redundant[NCS]; 
    struct ID *ID_redundant[NCS]; 
    struct IEN *IEN[MAX_LEVELS][NCS]; /* global at each level */
    struct FNODE *TWW[MAX_LEVELS][NCS];	/* for nodal averages */
    struct ID *ID[MAX_LEVELS][NCS];
    struct NEI NEI[MAX_LEVELS][NCS];
    struct SUBEL *EL[MAX_LEVELS][NCS];
    struct EG *elt_del[MAX_LEVELS][NCS];
    struct EK *elt_k[MAX_LEVELS][NCS];
    struct CC *cc[NCS];
    struct CCX *ccx[NCS];
    struct CC *CC[MAX_LEVELS][NCS];
    struct CCX *CCX[MAX_LEVELS][NCS];

    struct Crust crust;

    higher_precision *Eqn_k1[MAX_LEVELS][NCS],*Eqn_k2[MAX_LEVELS][NCS],*Eqn_k3[MAX_LEVELS][NCS];  
    int *Node_map [MAX_LEVELS][NCS];
    int *Node_eqn [MAX_LEVELS][NCS];
    int *Node_k_id[MAX_LEVELS][NCS];

    double *BI[MAX_LEVELS][NCS],*BPI[MAX_LEVELS][NCS];

    double *P[NCS],*F[NCS],*H[NCS],*S[NCS],*U[NCS];
    double *T[NCS],*Tdot[NCS],*buoyancy[NCS];
    double *u1[NCS];
    double *temp[NCS],*temp1[NCS];
    float *NP[NCS],*edot[NCS],*Mass[NCS],*tw[NCS];
    float *MASS[MAX_LEVELS][NCS];
    double *ZZ;
    double *SX[MAX_LEVELS][NCS][4],*X[MAX_LEVELS][NCS][4];
    double *sx[NCS][4],*x[NCS][4];
    double *surf_det[NCS][5];
    float *SinCos[MAX_LEVELS][NCS][4];
    float *TT;
    float *V[NCS][4],*GV[NCS][4],*GV1[NCS][4];

    float *stress[NCS];		
    float *Fas670[NCS],*Fas410[NCS],*Fas670_b[NCS],*Fas410_b[NCS];		
    float *Fascmb[NCS],*Fascmb_b[NCS];

    float *Vi[NCS],*EVi[NCS];
    float *VI[MAX_LEVELS][NCS],*EVI[MAX_LEVELS][NCS];
    float *TW[MAX_LEVELS][NCS];	/* nodal weightings */

    int num_zero_resid[MAX_LEVELS][NCS];	       
    int *zero_resid[MAX_LEVELS][NCS];	       
    int *surf_element[NCS],*surf_node[NCS];	       
    int *mat[NCS];	     
    float *VIP[NCS];
    unsigned int *ELEMENT[MAX_LEVELS][NCS],*NODE[MAX_LEVELS][NCS];
    unsigned int *element[NCS],*node[NCS];
    unsigned int *eqn[NCS],*EQN[MAX_LEVELS][NCS];

    float *age[NCS];	/* nodal weightings */
    float *age_t;
 		
    struct LM *lm[NCS];
    struct LM *LMD[MAX_LEVELS][NCS];
  
    struct Shape_function1 M; /* master-element shape funtions */
    struct Shape_function1_dx Mx; 
    struct Shape_function N;
    struct Shape_function NM;
    struct Shape_function_dx Nx;
    struct Shape_function1 L; /* master-element shape funtions */
    struct Shape_function1_dx Lx; 
    struct Shape_function_dx NMx;
 
  /* for temperature initial conditions */
  int number_of_perturbations;
  int perturb_ll[32];
  int perturb_mm[32];
  int load_depth[32];
  float perturb_mag[32];
  /*ccccc*/

    void (* build_forcing_term)(void*);
    void (* iterative_solver)(void*);
    void (* next_buoyancy_field)(void*);
    void (* obtain_gravity)(void*);
    void (* problem_settings)(void*);
    void (* problem_derived_values)(void*);
    void (* problem_allocate_vars)(void*);
    void (* problem_boundary_conds)(void*);
    void (* problem_node_positions)(void*);
    void (* problem_update_node_positions)(void*);
    void (* problem_initial_fields)(void*);
    void (* problem_tracer_setup)(void*);
    void (* problem_tracer_advection)(void*);
    void (* problem_tracer_output)(void*);
    void (* problem_update_bcs)(void*);
    void (* special_process_new_velocity)(void*);
    void (* special_process_new_buoyancy)(void*);
    void (* solve_stokes_problem)(void*); 
    void (* solver_allocate_vars)(void*); 
    void (* transform)(void*);

    float (* node_space_function[3])(void*);
 
};
