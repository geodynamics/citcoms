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

/* In this file we define the following data structures:
 *
 *  HDF5_INFO
 *    Used for collective output of citcom data.
 *
 *  HDF5_TIME
 *    Used to define table with timing information.
 *
 * Any required initialization steps are performed in h5output_open().
 *
 */

struct HDF5_TIME
{
    float time;
    float time_step;
    float cpu;
    float cpu_step;
};

struct HDF5_INFO
{
    char filename[100];

    /* Keep a reference to the open hdf5 output file */
    hid_t file_id;

    /* Default type used is H5T_NATIVE_FLOAT */
    hid_t type_id;

    /* Keep track of how many times we call h5output() */
    int count;

    /* Group names under which to store the appropriate data,
     * represented by an array of strings. For a regional
     * model, only cap_groups[0] should be used.
     */
    char cap_groups[12][7];
    
    /* Cap ID for current process */
    int capid;


    /* Temporary data buffers to use in dataset writes...
     * Note that most of these buffers correspond to time-slices
     * over a filespace in the HDF5 file.
     *
     * vector3d: coord, velocity
     * scalar3d: temperature, viscosity, pressure
     * vector2d: surf_coord, surf_velocity
     * scalar2d: surf_heatflux, surf_topography
     * scalar1d: horiz_avg_temperature, horiz_rms_vz, horiz_rms_vxy
     *
     */
    
    double *connectivity;           /* shape (nel,8) */
    double *material;               /* shape (nel,) */
    double *stress;                 /* shape (nx,ny,nz,6) */

    double *vector3d;               /* shape (nx,ny,nz,3) */
    double *scalar3d;               /* shape (nx,ny,nz) */
    double *vector2d;               /* shape (nx,ny,2) */
    double *scalar2d;               /* shape (nx,ny) */
    double *scalar1d;               /* shape (nz,) */
    
};
