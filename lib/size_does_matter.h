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

#if !defined(CitcomS_size_does_matter_h)
#define CitcomS_size_does_matter_h

struct All_variables;

void twiddle_thumbs(struct All_variables *yawn);
void construct_shape_function_derivatives(struct All_variables *E);
void get_rtf_at_vpts(struct All_variables *E, int m, int lev, int el, double rtf[4][9]);
void get_rtf_at_ppts(struct All_variables *E, int m, int lev, int el, double rtf[4][9]);
void construct_surf_det(struct All_variables *E);
void construct_bdry_det(struct All_variables *E);
void get_global_1d_shape_fn(struct All_variables *E, int el, struct Shape_function1 *GM, struct Shape_function1_dA *dGammax, int top, int m);
void get_global_1d_shape_fn_L(struct All_variables *E, int el, struct Shape_function1 *GM, struct Shape_function1_dA *dGammax, int top, int m);
void construct_c3x3matrix_el(struct All_variables *E, int el, struct CC *cc, struct CCX *ccx, int lev, int m, int pressure);
void construct_side_c3x3matrix_el(struct All_variables *E, int el, struct CC *cc, struct CCX *ccx, int lev, int m, int pressure, int side);
void mass_matrix(struct All_variables *E);

#endif
