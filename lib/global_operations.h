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

#if !defined(CitcomS_global_operations_h)
#define CitcomS_global_operations_h

struct All_variables;

void remove_horiz_ave2(struct All_variables *E, double **X);
void return_horiz_ave_f(struct All_variables *E, float **X, float *H);
double return_bulk_value_d(struct All_variables *E, double **Z, int average);
void sum_across_surface(struct All_variables *E, float *data, int total);
void sum_across_surf_sph1(struct All_variables *E, float *sphc, float *sphs);
double global_vdot(struct All_variables *E, double **A, double **B, int lev);
double global_pdot(struct All_variables *E, double **A, double **B, int lev);
double global_v_norm2(struct All_variables *E, double **V);
double global_p_norm2(struct All_variables *E, double **P);
double global_div_norm2(struct All_variables *E, double **A);
float global_fmin(struct All_variables *E, float a);
double Tmaxd(struct All_variables *E, double **T);
float Tmax(struct All_variables *E, float **T);
double vnorm_nonnewt(struct All_variables *E, double **dU, double **U, int lev);
void sum_across_depth_sph1(struct All_variables *E, float *sphc, float *sphs);
void broadcast_vertical(struct All_variables *E, float *sphc, float *sphs, int root);
void remove_rigid_rot(struct All_variables *E);

#endif
