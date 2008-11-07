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


/* 
 * This struct parameterizes those functions which are different
 * between the full and regional solvers.
 */

struct Solver {

    /* Boundary_conditions.c */
    void (*velocity_boundary_conditions)(struct All_variables *);
    void (*temperature_boundary_conditions)(struct All_variables *);

    /* Geometry_cartesian.c */
    void (*set_2dc_defaults)(struct All_variables *);
    void (*set_2pt5dc_defaults)(struct All_variables *);
    void (*set_3dc_defaults)(struct All_variables *);
    void (*set_3dsphere_defaults)(struct All_variables *);

    /* Lith_age.c */
    void (*lith_age_read_files)(struct All_variables *, int);

    /* Parallel_related.c */
    void (*parallel_processor_setup)(struct All_variables *);
    void (*parallel_domain_decomp0)(struct All_variables *);
    void (*parallel_domain_boundary_nodes)(struct All_variables *);
    void (*parallel_communication_routs_v)(struct All_variables *);
    void (*parallel_communication_routs_s)(struct All_variables *);
    void (*exchange_id_d)(struct All_variables *, double **, int);

    /* Read_input_from_files.c */
    void (*read_input_files_for_timesteps)(struct All_variables *, int, int);

    /* Version_dependent.c */
    void (*node_locations)(struct All_variables *);
    void (*construct_tic_from_input)(struct All_variables *);
    void (*construct_boundary)(struct All_variables *);

} solver;
