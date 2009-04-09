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

#if !defined(CitcomS_element_calculations_h)
#define CitcomS_element_calculations_h

struct All_variables;
struct Shape_function;
struct CC;
struct CCX;

typedef float higher_precision; /*XXX*/

void assemble_forces(struct All_variables *E, int penalty);
void assemble_forces_pseudo_surf(struct All_variables *E, int penalty);
void get_ba_p(struct Shape_function *N, struct Shape_function_dx *GNx, struct CC *cc, struct CCX *ccx, double rtf[4][9], int dims, double ba[9][9][4][7]);
void get_elt_k(struct All_variables *E, int el, double elt_k[24*24], int lev, int m, int iconv);
void assemble_del2_u(struct All_variables *E, double **u, double **Au, int level, int strip_bcs);
void e_assemble_del2_u(struct All_variables *E, double **u, double **Au, int level, int strip_bcs);
void n_assemble_del2_u(struct All_variables *E, double **u, double **Au, int level, int strip_bcs);
void build_diagonal_of_K(struct All_variables *E, int el, double elt_k[24*24], int level, int m);
void build_diagonal_of_Ahat(struct All_variables *E);
void assemble_c_u(struct All_variables *E, double **U, double **result, int level);
void assemble_div_rho_u(struct All_variables *E, double **U, double **result, int level);
void assemble_div_u(struct All_variables *E, double **U, double **divU, int level);
void assemble_grad_p(struct All_variables *E, double **P, double **gradP, int lev);
void get_elt_c(struct All_variables *E, int el, higher_precision elt_c[24][1], int lev, int m);
void get_elt_g(struct All_variables *E, int el, higher_precision elt_del[24][1], int lev, int m);
void get_elt_f(struct All_variables *E, int el, double elt_f[24], int bcs, int m);
void get_aug_k(struct All_variables *E, int el, double elt_k[24*24], int level, int m);

#endif
