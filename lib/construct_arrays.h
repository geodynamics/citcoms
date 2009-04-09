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

#if !defined(CitcomS_construct_arrays_h)
#define CitcomS_construct_arrays_h

struct All_variables;

void construct_ien(struct All_variables *E);
void construct_surface(struct All_variables *E);
void construct_id(struct All_variables *E);
void construct_lm(struct All_variables *E);
void construct_node_maps(struct All_variables *E);
void construct_masks(struct All_variables *E);
void construct_sub_element(struct All_variables *E);
void construct_elt_gs(struct All_variables *E);
void construct_elt_cs(struct All_variables *E);
void construct_stiffness_B_matrix(struct All_variables *E);
int layers_r(struct All_variables *E, float r);
int layers(struct All_variables *E, int m, int node);
void construct_mat_group(struct All_variables *E);

#endif
