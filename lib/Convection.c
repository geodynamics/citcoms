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
/* Assumes parameter list is opened and reads the things it needs.
   Variables are initialized etc, default values are set */


#include "global_defs.h"

void set_convection_defaults(struct All_variables *E)
{
  /*
    void PG_timestep();
    void read_convection_settings();
    void convection_derived_values();
    void convection_allocate_memory();
    void convection_boundary_conditions();
    void convection_initial_fields();
    void twiddle_thumbs();
  */
    E->control.CONVECTION = 1;

    E->advection.timestep = 0.0;
    E->advection.timesteps = 0;
    E->advection.temp_iterations = 2; /* petrov-galerkin iterations: minimum value. */
    E->advection.total_timesteps = 1;
    E->advection.sub_iterations = 1;
    E->advection.last_sub_iterations = 1;
    E->advection.gamma = 0.5;
    E->advection.dt_reduced = 1.0;

    E->monitor.T_maxvaried = 1.05;

    E->next_buoyancy_field = PG_timestep;
    E->next_buoyancy_field_init = PG_timestep_init;
    E->special_process_new_buoyancy = twiddle_thumbs;
    E->problem_settings = read_convection_settings;
    E->problem_derived_values = convection_derived_values;
    E->problem_allocate_vars = convection_allocate_memory;
    E->problem_boundary_conds = convection_boundary_conditions;
    E->problem_initial_fields = convection_initial_fields;
    E->problem_update_node_positions = twiddle_thumbs;
    E->problem_update_bcs = twiddle_thumbs;

    return;
}

void read_convection_settings(struct All_variables *E)
{
    /* parameters */

    advection_diffusion_parameters(E);

    return;
}

/* =================================================================
   Any setup which relates only to the convection stuff goes in here
   ================================================================= */

void convection_derived_values(struct All_variables *E)
{

  return;
}

void convection_allocate_memory(struct All_variables *E)

{ void advection_diffusion_allocate_memory();

  advection_diffusion_allocate_memory(E);

  return;
}

/* ============================================ */

void convection_initial_fields(struct All_variables *E)
{
    void convection_initial_temperature();

    convection_initial_temperature(E);

  return; }

/* =========================================== */

void convection_boundary_conditions(struct All_variables *E)
{
    (E->solver.velocity_boundary_conditions)(E);      /* universal */
    (E->solver.temperature_boundary_conditions)(E);
    return;
}

