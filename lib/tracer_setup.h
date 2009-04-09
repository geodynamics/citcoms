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

#if !defined(CitcomS_tracer_setup_h)
#define CitcomS_tracer_setup_h

struct All_variables;

void tracer_input(struct All_variables *E);
void tracer_initial_settings(struct All_variables *E);
void tracer_advection(struct All_variables *E);
void count_tracers_of_flavors(struct All_variables *E);
void initialize_tracers(struct All_variables *E);
void cart_to_sphere(struct All_variables *E, double x, double y, double z, double *theta, double *phi, double *rad);
void sphere_to_cart(struct All_variables *E, double theta, double phi, double rad, double *x, double *y, double *z);
void get_neighboring_caps(struct All_variables *E);
void allocate_tracer_arrays(struct All_variables *E, int j, int number_of_tracers);
void expand_tracer_arrays(struct All_variables *E, int j);
void expand_later_array(struct All_variables *E, int j);
int icheck_processor_shell(struct All_variables *E, int j, double rad);
int icheck_that_processor_shell(struct All_variables *E, int j, int nprocessor, double rad);

#endif
