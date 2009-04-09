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

#if !defined(CitcomS_instructions_h)
#define CitcomS_instructions_h

struct All_variables;

void initial_mesh_solver_setup(struct All_variables *E);
void read_instructions(struct All_variables *E, char *filename);
void initial_setup(struct All_variables *E);
void initialize_material(struct All_variables *E);
void initial_conditions(struct All_variables *E);
void check_settings_consistency(struct All_variables *E);
void global_default_values(struct All_variables *E);
void check_bc_consistency(struct All_variables *E);
void report(struct All_variables *E, char *string);
void record(struct All_variables *E, char *string);
void initial_pressure(struct All_variables *E);
void initial_velocity(struct All_variables *E);
void mkdatadir(const char *dir);
void output_finalize(struct All_variables *E);

#endif
