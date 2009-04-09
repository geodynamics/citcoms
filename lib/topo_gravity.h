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

#if !defined(CitcomS_topo_gravity_h)
#define CitcomS_topo_gravity_h

struct All_variables;

void get_STD_topo(struct All_variables *E, float **tpg, float **tpgb, float **divg, float **vort, int ii);
void get_STD_freesurf(struct All_variables *E, float **freesurf);
void allocate_STD_mem(struct All_variables *E, float **SXX, float **SYY, float **SZZ, float **SXY, float **SXZ, float **SZY, float **divv, float **vorv);
void free_STD_mem(struct All_variables *E, float **SXX, float **SYY, float **SZZ, float **SXY, float **SXZ, float **SZY, float **divv, float **vorv);
void compute_nodal_stress(struct All_variables *E, float **SXX, float **SYY, float **SZZ, float **SXY, float **SXZ, float **SZY, float **divv, float **vorv);
void compute_geoid(struct All_variables *E);
void get_CBF_topo(struct All_variables *E, float **H, float **HB);

#endif
