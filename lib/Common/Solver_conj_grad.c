#include "element_definitions.h"
#include "global_defs.h"

void set_cg_defaults(E)
     struct All_variables *E;
{ void assemble_forces_iterative();
  void solve_constrained_flow_iterative();
  void cg_allocate_vars();

  E->build_forcing_term = assemble_forces_iterative;
  E->solve_stokes_problem = solve_constrained_flow_iterative;
  E->solver_allocate_vars = cg_allocate_vars;


  return;
}

void cg_allocate_vars(E)
     struct All_variables *E;
{ 
  /* Nothing required ONLY by conj-grad stuff  */
 /* printf("here here\n"); */

  return;

}

void assemble_forces_iterative(E)
    struct All_variables *E;
{ 
  int i;

  void assemble_forces();
  void strip_bcs_from_residual();
    
  assemble_forces(E,0);

  strip_bcs_from_residual(E,E->F,E->mesh.levmax);

  return; 

}
