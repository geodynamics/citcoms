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

#if !defined(CitcomS_ggrd_handling_h)
#define CitcomS_ggrd_handling_h

/*

routines that deal with GMT/netcdf grd I/O as supported through
the ggrd subroutines of the hc package

*/

struct All_variables;

void ggrd_init_tracer_flavors(struct All_variables *);
void ggrd_full_temp_init(struct All_variables *);
void ggrd_reg_temp_init(struct All_variables *);
void ggrd_read_mat_from_file(struct All_variables *, int );
void ggrd_read_ray_from_file(struct All_variables *, int );
void ggrd_read_vtop_from_file(struct All_variables *, int );
void ggrd_read_age_from_file(struct All_variables *, int );
void ggrd_adjust_tbl_rayleigh(struct All_variables *,double **);

#endif
