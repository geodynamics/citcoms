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

#if !defined(CitcomS_regional_tracer_advection_h)
#define CitcomS_regional_tracer_advection_h

struct All_variables;

void regional_tracer_setup(struct All_variables *E);
int regional_iget_element(struct All_variables *E, int m, int iprevious_element, double dummy1, double dummy2, double dummy3, double theta, double phi, double rad);
int regional_icheck_cap(struct All_variables *E, int icap, double theta, double phi, double rad, double junk);
void regional_get_velocity(struct All_variables *E, int m, int nelem, double theta, double phi, double rad, double *velocity_vector);
void regional_keep_within_bounds(struct All_variables *E, double *x, double *y, double *z, double *theta, double *phi, double *rad);
void regional_lost_souls(struct All_variables *E);

#endif
