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

#if !defined(CitcomS_nodal_mesh_h)
#define CitcomS_nodal_mesh_h

struct All_variables;

void v_from_vector(struct All_variables *E);
void v_from_vector_pseudo_surf(struct All_variables *E);
void velo_from_element(struct All_variables *E, float VV[4][9], int m, int el, int sphere_key);
void velo_from_element_d(struct All_variables *E, double VV[4][9], int m, int el, int sphere_key);
void p_to_nodes(struct All_variables *E, double **P, float **PN, int lev);
void visc_from_gint_to_nodes(struct All_variables *E, float **VE, float **VN, int lev);
void visc_from_nodes_to_gint(struct All_variables *E, float **VN, float **VE, int lev);
void visc_from_gint_to_ele(struct All_variables *E, float **VE, float **VN, int lev);
void visc_from_ele_to_gint(struct All_variables *E, float **VN, float **VE, int lev);

#endif
