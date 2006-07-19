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
 * which is used for collective output of citcom data.
 */

struct HDF5_INFO {

    /* keep a reference to the open hdf5 file */
    hid_t   file_id;

    /* cap group names */
    char cap_group[12][8] = {
        "/cap00", "/cap01", "/cap02", "/cap03",
        "/cap04", "/cap05", "/cap06", "/cap07",
        "/cap08", "/cap09", "/cap10", "/cap11"
    };

    /* In order to create proper array hyperslabs for time-varying data,
     * keep track of which output cycle is about to begin.
     */
    int cycle;


    /* Temporary buffers to use in dataspace HDF5 API calls.
     * 
     * Note that for time-varying fields, the first dimension is time
     * (defined as extendible so that data can be appended). For
     * vector fields, the last dimension varies over the components
     * (which range over 0,1 or 0,1,2 or 0,1,2,3,4,5 depending on
     * the whether data is a 2D vector, 3D vector, or symmetric tensor)
     * 
     * The rank, dims[], and maxdims[] variables are used for
     * creating a dataspace (specifically, a filespace).
     *
     */

    /* for rank 5 arrays (time-varying 3D vector fields) */
    int rank5 = 5;
    hsize_t dims5[5];
    hsize_t maxdims5[5];

    /* for rank 4 arrays (time-varying 3D scalar fields) */
    int rank4 = 4;
    hsize_t dims4[4];
    hsize_t maxdims4[4];

    /* for rank 3 arrays (time-varying 2D scalar field,
     * fixed 3D scalar field, or simple 3D array) */
    int rank3 = 3;
    hsize_t dims3[3];
    hsize_t maxdims3[3];

    /* for rank 2 arrays (time-varying 1D array, or fixed 2D array) */
    int rank2 = 2;
    hsize_t dims2[2];
    hsize_t maxdims2[2];

    /* for rank 1 arrays */
    int rank1 = 1;
    hsize_t dims1[1];
    hsize_t maxdims1[1];
    

    /* Temporary buffers to use in HDF5 hyperslab API calls
     *
     * The count[], stride[], block[], and offset[] variables
     * are used for creating a hyperslab. The hyperslabs are
     * used to embed a memspace in the larger filespace so that
     * a global array in the file can be built from many local
     * ones in memory (per processor).
     */
    
    /* for rank 4 array hyperslabs */
    hsize_t count4[4];
    hsize_t stride4[4];
    hsize_t block4[4];
    hsize_t offset4[4];

    /* for rank 3 array hyperslabs */
    hsize_t count3[3];
    hsize_t stride3[3];
    hsize_t block3[3];
    hsize_t offset3[3];

    /* for rank 2 array hyperslabs */
    hsize_t count2[2];
    hsize_t stride2[2];
    hsize_t block2[2];
    hsize_t offset2[2];

    /* for rank 1 array hyperslabs */
    hsize_t count1[1];
    hsize_t stride1[1];
    hsize_t block1[1];
    hsize_t offset1[1];


    /* Temporary data buffers to use in dataset writes...
     * Note that these buffers correspond to a time-slice
     * over a memspace in the HDF5 API calls. 
     */
    double *connectivity;           // shape (mesh.nel,8)
    double *coord;                  // shape (nx,ny,nz,3)
    double *velocity;               // shape (nx,ny,nz,3)
    double *viscosity;              // shape (nx,ny,nz,3)
    double *stress;                 // shape (nx,ny,nz,6)
    double *pressure;               // shape (nx,ny,nz)
    double *temperature;            // shape (nx,ny,nz)
    double *surf_coord;             // shape (nx,ny,3)
    double *surf_heatflux;          // shape (nx,ny)
    double *surf_topography;        // shape (nx,ny)
    double *surf_velocity;          // shape (nx,ny,3)
    double *horiz_avg_temperature;  // shape (nz,)
    double *horiz_rms_vz;           // shape (nz,)
    double *horiz_rms_vxy;          // shape (nz,)
} hdf5;
