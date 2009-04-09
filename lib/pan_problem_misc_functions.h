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

#if !defined(CitcomS_pan_problem_misc_functions_h)
#define CitcomS_pan_problem_misc_functions_h

#include <stddef.h>
#include <stdio.h>

struct All_variables;

int get_process_identifier(void);
void unique_copy_file(struct All_variables *E, char *name, char *comment);
void apply_side_sbc(struct All_variables *E);
void get_buoyancy(struct All_variables *E, double **buoy);
int read_double_vector(FILE *in, int num_columns, double *fields);
double myatan(double y, double x);
double return1_test(void);
void xyz2rtp(float x, float y, float z, float *rout);
void xyz2rtpd(float x, float y, float z, double *rout);
void calc_cbase_at_node(int cap, int node, float *base, struct All_variables *E);
void convert_pvec_to_cvec(float vr, float vt, float vp, float *base, float *cvec);
void *safe_malloc(size_t size);
void myerror(struct All_variables *E, char *message);
void get_r_spacing_fine(double *rr, struct All_variables *E);
void get_r_spacing_at_levels(double *rr, struct All_variables *E);

#ifdef ALLOW_ELLIPTICAL
double theta_g(double , struct All_variables *);
#endif

#endif
