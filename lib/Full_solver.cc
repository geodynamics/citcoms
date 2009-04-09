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

#include "full_solver.h"

#include "global_defs.h"

#include "full_boundary_conditions.h"
#include "full_geometry_cartesian.h"
#include "full_lith_age_read_files.h"
#include "full_parallel_related.h"
#include "full_read_input_from_files.h"
#include "full_version_dependent.h"


void full_solver_init(struct All_variables *E)
{
    /* Boundary_conditions.c */
    E->solver.velocity_boundary_conditions = full_velocity_boundary_conditions;
    E->solver.temperature_boundary_conditions = full_temperature_boundary_conditions;

    /* Geometry_cartesian.c */
    E->solver.set_2dc_defaults = full_set_2dc_defaults;
    E->solver.set_2pt5dc_defaults = full_set_2pt5dc_defaults;
    E->solver.set_3dc_defaults = full_set_3dc_defaults;
    E->solver.set_3dsphere_defaults = full_set_3dsphere_defaults;

    /* Lith_age.c */
    E->solver.lith_age_read_files = full_lith_age_read_files;

    /* Parallel_related.c */
    E->solver.parallel_processor_setup = full_parallel_processor_setup;
    E->solver.parallel_domain_decomp0 = full_parallel_domain_decomp0;
    E->solver.parallel_domain_boundary_nodes = full_parallel_domain_boundary_nodes;
    E->solver.parallel_communication_routs_v = full_parallel_communication_routs_v;
    E->solver.parallel_communication_routs_s = full_parallel_communication_routs_s;
    E->solver.exchange_id_d = full_exchange_id_d;

    /* Read_input_from_files.c */
    E->solver.read_input_files_for_timesteps = full_read_input_files_for_timesteps;

    /* Version_dependent.c */
    E->solver.node_locations = full_node_locations;
    E->solver.construct_boundary = full_construct_boundary;
    
    return;
}


/* End of file */
