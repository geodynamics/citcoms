// petsc_citcoms.h
#ifndef __CitcomS__PETSc__h__
#define __CitcomS__PETSc__h__

#include <petscksp.h>

#ifdef __cplusplus
extern "C" {
#endif
  
struct MultiGrid_PC
{
  PC self;
  struct All_variables *E;
  // multigrid stuff
  double acc;
  int smooth_up;
  int smooth_down;
  int smooth_coarse;
  int smooth_fine;
  int max_vel_iterations;
  int cycle_type;
  double status;
  int level;
  int levmax;
  PetscBool mg_monitor;
  int nno;
  int caps_per_proc;
  double *V[NCS];
  double *RR[NCS];
};

struct MatMultShell_del2_u
{
  struct All_variables *E;
  int level;
  int neq;
  int nel;
  double *u[NCS];
  double *Ku[NCS];
};

struct MatMultShell_grad_p
{
  struct All_variables *E;
  int level;
  int neq;
  int nel;
  double *p[NCS];
  double *Gp[NCS];
};

struct MatMultShell_div_u
{
  struct All_variables *E;
  int level;
  int neq;
  int nel;
  double *u[NCS];
  double *Du[NCS];
};

struct MatMultShell_div_rho_u
{
  struct All_variables *E;
  int level;
  int neq;
  int npno;
  double *u[NCS];
  double *Du[NCS];
};

void strip_bcs_from_residual_PETSc( 
    struct All_variables *E, Vec Res, int level );
PetscErrorCode initial_vel_residual_PETSc( struct All_variables *E,
                                 Vec V, Vec P, Vec F,
                                 double acc );
double global_v_norm2_PETSc( struct All_variables *E, Vec v );
double global_p_norm2_PETSc( struct All_variables *E, Vec p );
double global_div_norm2_PETSc( struct All_variables *E,  Vec a );
PetscErrorCode assemble_c_u_PETSc( struct All_variables *E, Vec U, Vec res, int level );

PetscErrorCode PC_Apply_MultiGrid( PC pc, Vec x, Vec y );

PetscErrorCode MatShellMult_del2_u( Mat K, Vec U, Vec KU );
PetscErrorCode MatShellMult_grad_p( Mat G, Vec P, Vec GP );
PetscErrorCode MatShellMult_div_u( Mat D, Vec U, Vec DU );
PetscErrorCode MatShellMult_div_rho_u( Mat D, Vec U, Vec DU );

#ifdef __cplusplus
}
#endif

#endif // __CitcomS__PETSc__h__
