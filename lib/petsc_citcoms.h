// petsc_citcoms.h
#ifndef __CitcomS__PETSc__h__
#define __CitcomS__PETSc__h__

#include <petscksp.h>

#ifdef __cplusplus
extern "C" {
#endif

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
