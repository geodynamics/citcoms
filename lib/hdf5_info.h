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

/* In this file we define the contents of the HDF5_INFO data structure
 * that is used for collective output of citcom data. The contents
 * of this structure are initialized by the function h5output_open().
 */

struct HDF5_INFO
{
    /* Keep a reference to the open hdf5 output file */
    hid_t   file_id;

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
     */
    
    double *connectivity;           /* shape (nel,8) */
    double *material;               /* shape (nel,) */

    double *coord;                  /* shape (nx,ny,nz,3) */
    double *velocity;               /* shape (nx,ny,nz,3) */
    double *temperature;            /* shape (nx,ny,nz) */
    double *viscosity;              /* shape (nx,ny,nz) */
    double *pressure;               /* shape (nx,ny,nz) */
    double *stress;                 /* shape (nx,ny,nz,6) */

    double *surf_coord;             /* shape (nx,ny,3) */
    double *surf_velocity;          /* shape (nx,ny,3) */
    double *surf_heatflux;          /* shape (nx,ny) */
    double *surf_topography;        /* shape (nx,ny) */
    
    double *horiz_avg_temperature;  /* shape (nz,) */
    double *horiz_rms_vz;           /* shape (nz,) */
    double *horiz_rms_vxy;          /* shape (nz,) */

} hdf5;
