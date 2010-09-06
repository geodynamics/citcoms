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
/*

routines that deal with GMT/netcdf grd I/O as supported through
the ggrd subroutines of the hc package

*/
void ggrd_init_tracer_flavors(struct All_variables *);
int layers_r(struct All_variables *,float );
void myerror(struct All_variables *,char *);
void ggrd_full_temp_init(struct All_variables *);
void ggrd_reg_temp_init(struct All_variables *);
void ggrd_temp_init_general(struct All_variables *,int);
void ggrd_read_mat_from_file(struct All_variables *, int );
void ggrd_read_ray_from_file(struct All_variables *, int );
void ggrd_read_vtop_from_file(struct All_variables *, int);
void ggrd_read_age_from_file(struct All_variables *, int );
void xyz2rtp(float ,float ,float ,float *);
void xyz2rtpd(float ,float ,float ,double *);
float find_age_in_MY();
void ggrd_adjust_tbl_rayleigh(struct All_variables *,double **);
void ggrd_read_anivisc_from_file(struct All_variables *, int );

#define GGRD_DENS_MIN 3200	/* minimum density for PREM scaling of input files */
