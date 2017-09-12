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
#include "element_definitions.h"
#include "global_defs.h"

void set_cg_defaults(E)
     struct All_variables *E;
{ void assemble_forces_iterative();
  void solve_constrained_flow_iterative();
  void cg_allocate_vars();

  E->control.CONJ_GRAD = 1;
  E->build_forcing_term =   assemble_forces_iterative;
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
