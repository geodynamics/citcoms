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
struct ADVECTION {
  int ADVECTION;
 
  float gamma;
  float timestep;
  float diff_timestep;
  float fine_tune_dt;
  float dt_reduced;
  float fixed_timestep;
  float max_dimensionless_time;
 
  int min_timesteps;  
  int max_timesteps;
  int max_total_timesteps;
  int timesteps;
  int total_timesteps;
  int temp_iterations;
  int max_substeps;
  int sub_iterations;
  int last_sub_iterations; 

  float vel_substep_aggression;
  float temp_updatedness;
  float visc_updatedness;

  float lid_defining_velocity;
  float sub_layer_sample_level;

 
 } advection;


