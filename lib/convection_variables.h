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


struct CONVECTION { /* information controlling convection problems */

    int tic_method;
    float half_space_age;

#define PERTURB_MAX_LAYERS 255
    int number_of_perturbations;
    int perturb_ll[PERTURB_MAX_LAYERS];
    int perturb_mm[PERTURB_MAX_LAYERS];
    int load_depth[PERTURB_MAX_LAYERS];
    float perturb_mag[PERTURB_MAX_LAYERS];

  float blob_center[3];
  float blob_radius;
  float blob_dT;

    struct SOURCES {
	    int number;
	    float t_offset;
	    float Q[10];
	    float lambda[10];
	}  heat_sources;


#ifdef USE_GGRD
  /* for temperature init from grd files */
  int ggrd_tinit,ggrd_tinit_scale_with_prem;
  int ggrd_tinit_override_tbc,ggrd_tinit_limit_trange;
  double ggrd_tinit_scale,ggrd_tinit_offset,ggrd_vstage_transition;
  char ggrd_tinit_gfile[1000];
  char ggrd_tinit_dfile[1000];
  struct ggrd_gt ggrd_tinit_d[1];
  struct ggrd_t ggrd_time_hist;
  struct prem_model prem; 
#endif

} convection;


