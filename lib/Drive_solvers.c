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
#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "drive_solvers.h"

#ifdef USE_PETSC
#include "petsc_citcoms.h"
#endif

double global_vdot();
double vnorm_nonnewt();
int need_visc_update(struct All_variables *);
int need_to_iterate(struct All_variables *);


/************************************************************/

void general_stokes_solver_setup(struct All_variables *E)
{
  int i, m, lev, npno, neq, nel;
  void construct_node_maps();

  if (E->control.NMULTIGRID || E->control.NASSEMBLE)
    construct_node_maps(E);
  else
    for (i=E->mesh.gridmin;i<=E->mesh.gridmax;i++)
      E->elt_k[i][CPPR]=(struct EK *)malloc((E->lmesh.NEL[i]+1)*sizeof(struct EK));

#ifdef USE_PETSC
  /*
   * PETSc related initialization
   */
  lev = E->mesh.levmax;
  npno = E->lmesh.npno;
  neq = E->lmesh.neq;
  nel = E->lmesh.nel;

  E->mtx_del2_u.E = E;
  E->mtx_del2_u.level = E->mesh.levmax;
  E->mtx_del2_u.iSize = E->lmesh.neq;
  E->mtx_del2_u.oSize = E->lmesh.nel;

  E->mtx_grad_p.E = E;
  E->mtx_grad_p.level = E->mesh.levmax;
  E->mtx_grad_p.iSize = E->lmesh.nel;
  E->mtx_grad_p.oSize = E->lmesh.neq;
  
  E->mtx_div_u.E = E;
  E->mtx_div_u.level = E->mesh.levmax;
  E->mtx_div_u.iSize = E->lmesh.neq;
  E->mtx_div_u.oSize = E->lmesh.nel; 

  E->mtx_div_rho_u.E = E;
  E->mtx_div_rho_u.level = E->mesh.levmax;
  E->mtx_div_rho_u.iSize = E->lmesh.neq;
  E->mtx_div_rho_u.oSize = E->lmesh.npno;

  E->pcshell_ctx.self = E->pc;
  E->pcshell_ctx.E = E;
  E->pcshell_ctx.nno = E->lmesh.NEQ[lev];
  E->pcshell_ctx.acc = 1e-6;
  E->pcshell_ctx.level = lev;

  PetscMalloc( neq*sizeof(double), &E->mtx_del2_u.iData );
  PetscMalloc( neq*sizeof(double), &E->mtx_del2_u.oData );
  PetscMalloc( nel*sizeof(double), &E->mtx_grad_p.iData );
  PetscMalloc( neq*sizeof(double), &E->mtx_grad_p.oData );
  PetscMalloc( neq*sizeof(double), &E->mtx_div_u.iData );
  PetscMalloc( nel*sizeof(double), &E->mtx_div_u.oData );
  PetscMalloc( neq*sizeof(double), &E->mtx_div_rho_u.iData );
  PetscMalloc( npno*sizeof(double), &E->mtx_div_rho_u.oData );
  PetscMalloc( (E->lmesh.NEQ[lev])*sizeof(double), &E->pcshell_ctx.V );
  PetscMalloc( (E->lmesh.NEQ[lev])*sizeof(double), &E->pcshell_ctx.RR );

  // Create a Matrix shell for the K matrix
  MatCreateShell( PETSC_COMM_WORLD, 
			            neq, neq,
			            PETSC_DETERMINE, PETSC_DETERMINE,
			            (void *)&E->mtx_del2_u, 
			            &E->K );
  // associate a matrix multiplication operation to this shell
  MatShellSetOperation(E->K, MATOP_MULT, (void(*)(void))MatShellMult_del2_u);

  // Create a Matrix shell for the G matrix
  MatCreateShell( PETSC_COMM_WORLD, 
			            neq, nel,
			            PETSC_DETERMINE, PETSC_DETERMINE,
			            (void *)&E->mtx_grad_p, 
			            &E->G );
  // associate a matrix multiplication operation to this shell
  MatShellSetOperation(E->G, MATOP_MULT, (void(*)(void))MatShellMult_grad_p);

  // Create a Matrix shell for the D matrix
  MatCreateShell( PETSC_COMM_WORLD, 
			            nel, neq,
			            PETSC_DETERMINE, PETSC_DETERMINE,
			            (void *)&E->mtx_div_u, 
			            &E->D );
  // associate a matrix multiplication operation to this shell
  MatShellSetOperation(E->D, MATOP_MULT, (void(*)(void))MatShellMult_div_u);

  // Create a Matrix shell for the DC matrix
  MatCreateShell( PETSC_COMM_WORLD, 
			            npno, neq,
			            PETSC_DETERMINE, PETSC_DETERMINE,
			            (void *)&E->mtx_div_rho_u, 
			            &E->DC );
  // associate a matrix multiplication operation to this shell
  MatShellSetOperation(E->DC,MATOP_MULT,(void(*)(void))MatShellMult_div_rho_u);

  // Create and setup the KSP solver
  KSPCreate( PETSC_COMM_WORLD, &E->ksp );
  KSPSetOperators( E->ksp, E->K, E->K);// SAME_NONZERO_PATTERN ); 
  KSPSetFromOptions( E->ksp );
  
  if(strcmp(E->control.SOLVER_TYPE, "multigrid") == 0) 
  {
    KSPGetPC( E->ksp, &E->pc );
    PCSetType( E->pc, PCSHELL );
    PCShellSetContext( E->pc, (void *)&E->pcshell_ctx );
    PCShellSetApply( E->pc, PC_Apply_MultiGrid );
  }

  /************************************************************
   * End PETSc Initialization
   ************************************************************/
#endif
}

#ifdef USE_PETSC
void general_stokes_solver_teardown(struct All_variables *E)
{
  KSPDestroy(&E->ksp);

  MatDestroy(&E->K);
  MatDestroy(&E->G);
  MatDestroy(&E->D);
  MatDestroy(&E->DC);

  PetscFree( E->mtx_del2_u.iData );
  PetscFree( E->mtx_del2_u.oData );
  PetscFree( E->mtx_grad_p.iData );
  PetscFree( E->mtx_grad_p.oData );
  PetscFree( E->mtx_div_u.iData );
  PetscFree( E->mtx_div_u.oData );
  PetscFree( E->mtx_div_rho_u.iData );
  PetscFree( E->mtx_div_rho_u.oData );
  PetscFree( E->pcshell_ctx.V );
  PetscFree( E->pcshell_ctx.RR );
}
#endif

void general_stokes_solver(struct All_variables *E)
{
  void solve_constrained_flow_iterative();
  void construct_stiffness_B_matrix();
  void velocities_conform_bcs();
  void assemble_forces();
  void sphere_harmonics_layer();
  void get_system_viscosity();
  void remove_rigid_rot();

  double Udot_mag, dUdot_mag;
  int m,i;

  double *oldU, *delta_U;

  const int neq = E->lmesh.neq;

  E->monitor.visc_iter_count = 0; /* first solution */

  velocities_conform_bcs(E,E->U);

  assemble_forces(E,0);
  if(need_visc_update(E)){
    get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);
    construct_stiffness_B_matrix(E);
  } 
  
  solve_constrained_flow_iterative(E);

  
  if (need_to_iterate(E)) {
    /* outer iterations for velocity dependent viscosity */

      delta_U = (double *)malloc(neq*sizeof(double));
      oldU = (double *)malloc(neq*sizeof(double));
      for(i=0;i<neq;i++)
        oldU[i]=0.0;

    Udot_mag=dUdot_mag=0.0;

    E->monitor.visc_iter_count++;
    while (1) {    
     

	for (i=0;i<neq;i++) {
	  delta_U[i] = E->U[i] - oldU[i];
	  oldU[i] = E->U[i];
	}

      Udot_mag  = sqrt(global_vdot(E,oldU,oldU,E->mesh.levmax));
      dUdot_mag = vnorm_nonnewt(E,delta_U,oldU,E->mesh.levmax);


      if(E->parallel.me==0){
	fprintf(stderr,"Stress dep. visc./plast.: DUdot = %.4e (%.4e) for iteration %d\n",
		dUdot_mag,Udot_mag,E->monitor.visc_iter_count);
	fprintf(E->fp,"Stress dep. visc./plast.: DUdot = %.4e (%.4e) for iteration %d\n",
		dUdot_mag,Udot_mag,E->monitor.visc_iter_count);
	fflush(E->fp);
      }
      if ((E->monitor.visc_iter_count > 50) || 
	  (dUdot_mag < E->viscosity.sdepv_misfit))
	break;
      
      get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);
      construct_stiffness_B_matrix(E);
      solve_constrained_flow_iterative(E);
      
      E->monitor.visc_iter_count++;

    } /*end while*/

    free((void *) oldU);
    free((void *) delta_U);

  } /*end if we need iterations */

  /* remove the rigid rotation component from the velocity solution */
  if((E->sphere.caps == 12) &&
     (E->control.remove_rigid_rotation || E->control.remove_angular_momentum)) {
      remove_rigid_rot(E);
  }
}

int need_visc_update(struct All_variables *E)
{
  if(E->viscosity.update_allowed){
    /* always update */
    return 1;
  }else{
    /* deal with first time called */
    if(E->control.restart){	
      /* restart step - when this function is called, the cycle has
	 already been incremented */
      if(E->monitor.solution_cycles ==  E->monitor.solution_cycles_init + 1)
	return 1;
      else
	return 0;
    }else{
      /* regular step */
      if(E->monitor.solution_cycles == 0)
	return 1;
      else
	return 0;
    }
  }
}
int need_to_iterate(struct All_variables *E){
  if((E->control.force_iteration) && (E->monitor.solution_cycles == 0)){
    return 1;
  }else{
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
  /* anisotropic viscosity */
  if(E->viscosity.allow_anisotropic_viscosity){
    if(E->viscosity.anivisc_start_from_iso) /* first step will be
					       solved isotropically at
					       first  */
      return 1;
    else
      return (E->viscosity.SDEPV || E->viscosity.PDEPV)?(1):(0);
  }else{
#endif
  /* regular operation */
  return ((E->viscosity.SDEPV || E->viscosity.PDEPV)?(1):(0));
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
  }
#endif
  }
}
void general_stokes_solver_pseudo_surf(struct All_variables *E)
{
  void solve_constrained_flow_iterative();
  void construct_stiffness_B_matrix();
  void velocities_conform_bcs();
  void assemble_forces();
  void get_system_viscosity();
  void std_timestep();
  void remove_rigid_rot();
  void get_STD_freesurf(struct All_variables *, float**);

  double Udot_mag, dUdot_mag;
  int m,count,i;

  double *oldU, *delta_U;

  const int neq = E->lmesh.neq;

  velocities_conform_bcs(E,E->U);

  E->monitor.stop_topo_loop = 0;
  E->monitor.topo_loop = 0;
  if(E->monitor.solution_cycles==0) std_timestep(E);
  while(E->monitor.stop_topo_loop == 0) {
	  assemble_forces(E,0);
	  if(need_visc_update(E)){
	    get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);
	    construct_stiffness_B_matrix(E);
	  }
	  solve_constrained_flow_iterative(E);

	  if (E->viscosity.SDEPV || E->viscosity.PDEPV) {

			  delta_U = (double *)malloc(neq*sizeof(double));
			  oldU = (double *)malloc(neq*sizeof(double));
			  for(i=0;i<neq;i++)
				  oldU[i]=0.0;

		  Udot_mag=dUdot_mag=0.0;
		  count=1;

		  while (1) {

				  for (i=0;i<neq;i++) {
					  delta_U[i] = E->U[i] - oldU[i];
					  oldU[i] = E->U[i];
				  }

			  Udot_mag  = sqrt(global_vdot(E,oldU,oldU,E->mesh.levmax));
			  dUdot_mag = vnorm_nonnewt(E,delta_U,oldU,E->mesh.levmax);

			  if(E->parallel.me==0){
                              fprintf(stderr,"Stress dep. visc./plast.: DUdot = %.4e (%.4e) for iteration %d\n",
                                      dUdot_mag,Udot_mag,count);
                              fprintf(E->fp,"Stress dep. visc./plast.: DUdot = %.4e (%.4e) for iteration %d\n",
                                      dUdot_mag,Udot_mag,count);
				  fflush(E->fp);
			  }

			  if (count>50 || dUdot_mag<E->viscosity.sdepv_misfit)
				  break;

			  get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);
			  construct_stiffness_B_matrix(E);
			  solve_constrained_flow_iterative(E);

			  count++;

		  } /*end while */
      free((void *) oldU);
      free((void *) delta_U);

	  } /*end if SDEPV or PDEPV */
	  E->monitor.topo_loop++;
  }

  /* remove the rigid rotation component from the velocity solution */
  if((E->sphere.caps == 12) &&
     (E->control.remove_rigid_rotation || E->control.remove_angular_momentum)) {
      remove_rigid_rot(E);
  }

  get_STD_freesurf(E,E->slice.freesurf);
}
