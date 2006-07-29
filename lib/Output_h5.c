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

/* Routines to write the output of the finite element cycles into an
 * HDF5 file.
 */


#include <stdlib.h>
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "output_h5.h"

#ifdef USE_HDF5
#include "pytables.h"
#endif


/****************************************************************************
 * Function prototypes                                                      *
 ****************************************************************************/

void h5output_coord(struct All_variables *);
void h5output_velocity(struct All_variables *, int);
void h5output_temperature(struct All_variables *, int);
void h5output_viscosity(struct All_variables *, int);
void h5output_pressure(struct All_variables *, int);
void h5output_stress(struct All_variables *, int);
void h5output_material(struct All_variables *);
void h5output_tracer(struct All_variables *, int);
void h5output_surf_botm(struct All_variables *, int);
void h5output_surf_botm_pseudo_surf(struct All_variables *, int);
void h5output_ave_r(struct All_variables *, int);

#ifdef USE_HDF5
static hid_t h5create_group(hid_t loc_id,
                            const char *name,
                            size_t size_hint);

static void h5create_array(hid_t loc_id,
                           const char *name,
                           hid_t type_id,
                           int rank,
                           hsize_t *dims,
                           hsize_t *maxdims,
                           hsize_t *chunk_dims);

static void h5create_field(hid_t loc_id,
                           const char *name,
                           hid_t type_id,
                           int tdim, int xdim, int ydim, int zdim,
                           int components);

static void h5create_coord(hid_t loc_id,
                           hid_t type_id,
                           int nodex, int nodey, int nodez);

static void h5create_velocity(hid_t loc_id,
                              hid_t type_id,
                              int nodex, int nodey, int nodez);

static void h5create_stress(hid_t loc_id,
                            hid_t type_id,
                            int nodex, int nodey, int nodez);

static void h5create_temperature(hid_t loc_id,
                                 hid_t type_id,
                                 int nodex, int nodey, int nodez);

static void h5create_viscosity(hid_t loc_id,
                               hid_t type_id,
                               int nodex, int nodey, int nodez);

static void h5create_pressure(hid_t loc_id,
                              hid_t type_id,
                              int nodex, int nodey, int nodez);

static void h5create_surf_coord(hid_t loc_id,
                                hid_t type_id,
                                int nodex, int nodey);

static void h5create_surf_velocity(hid_t loc_id,
                                   hid_t type_id,
                                   int nodex, int nodey);

static void h5create_surf_heatflux(hid_t loc_id,
                                   hid_t type_id,
                                   int nodex, int nodey);

static void h5create_surf_topography(hid_t loc_id,
                                     hid_t type_id,
                                     int nodex, int nodey);

#endif

extern void parallel_process_termination();
extern void heat_flux(struct All_variables *);
extern void get_STD_topo(struct All_variables *, float**, float**,
                         float**, float**, int);


/****************************************************************************
 * Functions that control which data is saved to output file(s).            *
 * These represent possible choices for (E->output) function pointer.       *
 ****************************************************************************/

void h5output(struct All_variables *E, int cycles)
{
    if (cycles == 0) {
        h5output_coord(E);
        h5output_material(E);
    }

    h5output_velocity(E, cycles);
    h5output_temperature(E, cycles);
    h5output_viscosity(E, cycles);

    if(E->control.pseudo_free_surf)
    {
        if(E->mesh.topvbc == 2)
            h5output_surf_botm_pseudo_surf(E, cycles);
    }
    else
        h5output_surf_botm(E, cycles);

    if(E->control.tracer==1)
        h5output_tracer(E, cycles);

    //h5output_stress(E, cycles);
    //h5output_pressure(E, cycles);

    /* disable horizontal average h5output   by Tan2 */
    /* h5output_ave_r(E, cycles); */

    return;
}

void h5output_pseudo_surf(struct All_variables *E, int cycles)
{

    if (cycles == 0)
    {
        h5output_coord(E);
        h5output_material(E);
    }

    h5output_velocity(E, cycles);
    h5output_temperature(E, cycles);
    h5output_viscosity(E, cycles);
    h5output_surf_botm_pseudo_surf(E, cycles);

    if(E->control.tracer==1)
        h5output_tracer(E, cycles);

    //h5output_stress(E, cycles);
    //h5output_pressure(E, cycles);

    /* disable horizontal average h5output   by Tan2 */
    /* h5output_ave_r(E, cycles); */

    return;
}


/****************************************************************************
 * Functions to initialize and finalize access to HDF5 output file.         *
 * Responsible for creating all necessary groups, attributes, and arrays.   *
 ****************************************************************************/

/* This function should open the HDF5 file, and create any necessary groups,
 * arrays, and attributes. It should also initialize the hyperslab parameters
 * that will be used for future dataset writes
 */
void h5output_open(struct All_variables *E)
{
#ifdef USE_HDF5

    hid_t file_id;                  /* HDF5 file identifier */
    hid_t fapl_id;                  /* file access property list identifier */

    char *cap_name;
    hid_t cap_group;                /* group identifier for caps */
    hid_t surf_group, botm_group;   /* group identifier for cap surfaces */

    hid_t type_id;

    herr_t status;

    MPI_Comm comm = E->parallel.world; 
    MPI_Info info = MPI_INFO_NULL;

    int cap;
    int caps = E->sphere.caps;

    int p, px, py, pz;
    int nprocx, nprocy, nprocz;

    int nodex, nodey, nodez;


    /* set up file access property list with parallel I/O access */
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    status = H5Pset_fapl_mpio(fapl_id, comm, info);

    /* create a new file collectively and release property list identifier */
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    /* save the file identifier for later use */
    E->hdf5.file_id = file_id;

    /* determine current cap id and remember index */
    p = E->parallel.me;
    px = E->parallel.me_loc[1];
    py = E->parallel.me_loc[2];
    pz = E->parallel.me_loc[3];

    nprocx = E->parallel.nprocx;
    nprocy = E->parallel.nprocy;
    nprocz = E->parallel.nprocz;
    E->hdf5.capid = p/(nprocx*nprocy*nprocz);

    /* determine dimensions of mesh */
    nodex = E->mesh.nox;
    nodey = E->mesh.noy;
    nodez = E->mesh.noz;

    /* type to use for creating the fields */
    type_id = H5T_NATIVE_FLOAT;

    /* Create necessary groups and arrays */
    for(cap = 0; cap < caps; cap++)
    {
        /********************************************************************
         * Create /cap/ group                                               *
         ********************************************************************/

        cap_name = E->hdf5.cap_groups[cap];
        snprintf(cap_name, (size_t)7, "/cap%02d", cap);
        cap_group = h5create_group(file_id, cap_name, (size_t)0);

        h5create_coord(cap_group, type_id, nodex, nodey, nodez);
        h5create_velocity(cap_group, type_id, nodex, nodey, nodez);
        h5create_temperature(cap_group, type_id, nodex, nodey, nodez);
        h5create_viscosity(cap_group, type_id, nodex, nodey, nodez);
        h5create_pressure(cap_group, type_id, nodex, nodey, nodez);
        h5create_stress(cap_group, type_id, nodex, nodey, nodez);

        /********************************************************************
         * Create /cap/surf/ group                                          *
         ********************************************************************/

        surf_group = h5create_group(cap_group, "surf", (size_t)0);

        h5create_surf_coord(surf_group, type_id, nodex, nodey);
        h5create_surf_velocity(surf_group, type_id, nodex, nodey); 
        h5create_surf_heatflux(surf_group, type_id, nodex, nodey);
        h5create_surf_topography(surf_group, type_id, nodex, nodey);
        status = H5Gclose(surf_group);


        /********************************************************************
         * Create /cap/botm/ group                                          *
         ********************************************************************/

        botm_group = h5create_group(cap_group, "botm", (size_t)0);

        h5create_surf_coord(botm_group, type_id, nodex, nodey);
        h5create_surf_velocity(botm_group, type_id, nodex, nodey); 
        h5create_surf_heatflux(botm_group, type_id, nodex, nodey);
        h5create_surf_topography(botm_group, type_id, nodex, nodey);
        status = H5Gclose(botm_group);

        /* release resources */
        status = H5Gclose(cap_group);
    }
#endif
}

void h5output_close(struct All_variables *E)
{
#ifdef USE_HDF5
    herr_t status = H5Fclose(E->hdf5.file_id);
    /* XXX: free allocated memory too, if any */
#endif
}



/****************************************************************************
 * Private functions to simplify certain tasks in h5output_open()           *
 ****************************************************************************/
#ifdef USE_HDF5

static hid_t h5create_group(hid_t loc_id, const char *name, size_t size_hint)
{
    hid_t group_id;
    
    /* TODO:
     *  Make sure this function is called with an appropriately
     *  estimated size_hint parameter
     */
    group_id = H5Gcreate(loc_id, name, size_hint);
   
   /* TODO: Here, write any necessary attributes for PyTables compatibility. */

   return group_id;

}

static void h5create_array(hid_t loc_id,
                           const char *name,
                           hid_t type_id,
                           int rank,
                           hsize_t *dims,
                           hsize_t *maxdims,
                           hsize_t *chunk_dims)
{
    hid_t filespace;    /* file dataspace identifier */
    hid_t dcpl_id;      /* dataset creation property list identifier */
    hid_t dataset;      /* dataset identifier */

    herr_t status;

    /* create the dataspace for the dataset */
    filespace = H5Screate_simple(rank, dims, maxdims);
    
    dcpl_id = H5P_DEFAULT;
    if (chunk_dims != NULL)
    {
        /* modify dataset creation properties to enable chunking */
        dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
        status = H5Pset_chunk(plist, rank, chunk_dims);
        //status = H5Pset_fill_value(dcpl_id, H5T_NATIVE_FLOAT, &fillvalue);
    }

    /* create the dataset */
    dataset = H5Dcreate(loc_id, name, type_id, filespace, dcpl_id);

    /* TODO
     *  1. Add attribute strings necessary for PyTables compatibility
     */

    /* release resources */
    status = H5Pclose(dcpl_id);
    status = H5Sclose(filespace);
    status = H5Dclose(dataset);
}

static void h5write_array_hyperslab(hid_t dset_id,
                                    hid_t mem_type_id,
                                    const void *data,
                                    int rank,
                                    hsize_t *size,
                                    hsize_t *memdims,
                                    hsize_t *offset,
                                    hsize_t *stride,
                                    hsize_t *count,
                                    hsize_t *block)
{
    hid_t dataset;
    hid_t filespace;
    hid_t memspace;
    hid_t dxpl_id;

    herr_t status;
    
    /* extend the dataset if necessary */
    if(size != NULL)
    {
        status = H5Dextend(dataset, size);
    }

    /* get file dataspace */
    filespace = H5Dget_space(dataset);

    /* dataset transfer property list */
    dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    status  = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);
    // status = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_INDEPENDENT);

    /* create memory dataspace */
    memspace = H5Screate_simple(rank, memdims, NULL);

    /* hyperslab selection */
    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                 offset, stride, count, block);

    /* write the data to the hyperslab */
    status = H5Dwrite(dataset, mem_type_id, memspace, filespace, dxpl_id, data);
    
    /* release resources */
    status = H5Pclose(dxpl_id);
    status = H5Sclose(memspace);
    status = H5Sclose(filespace);
}

static void h5create_field(hid_t loc_id,
                           const char *name,
                           hid_t type_id,
                           int tdim, int xdim, int ydim, int zdim, int cdim)
{
    int rank = 0;
    hsize_t dims[5] = {0,0,0,0,0};
    hsize_t maxdims[5] = {0,0,0,0,0};
    hsize_t chunkdims[5] = {0,0,0,0,0};
    
    if (time > 0)
    {
        /* time-varying dataset -- determine spatial dimensionality */
        
        if ((xdim > 0) && (ydim > 0) && (zdim > 0))
        {
            rank = 1 + 3;
            
            dims[0] = 0;
            dims[1] = xdim;
            dims[2] = ydim;
            dims[3] = zdim;

            maxdims[0] = H5S_UNLIMITED;
            maxdims[1] = xdim;
            maxdims[2] = ydim;
            maxdims[3] = zdim;

            chunkdims[0] = 1;
            chunkdims[1] = xdim;
            chunkdims[2] = ydim;
            chunkdims[3] = zdim;
        }
        else if ((xdim > 0) && (ydim > 0))
        {
            rank = 1 + 2;

            dims[0] = 0;
            dims[1] = xdim;
            dims[2] = ydim;

            maxdims[0] = H5S_UNLIMITED;
            maxdims[1] = xdim;
            maxdims[2] = ydim;

            chunkdims[0] = 1;
            chunkdims[1] = xdim;
            chunkdims[2] = ydim;
        }
        else if (zdim > 0)
        {
            rank = 1 + 1;

            dims[0] = 0;
            dims[1] = zdim;

            maxdims[0] = H5S_UNLIMITED;
            maxdims[1] = zdim;

            chunkdims[0] = 1;
            chunkdims[1] = zdim;
        }

        /* if field has components, update last dimension */
        if (cdim > 0)
        {
            rank += 1;
            dims[rank-1] = cdim;
            maxdims[rank-1] = cdim;
            chunkdims[rank-1] = cdim;
        }
        /* finally, create the array */
        h5create_array(loc_id, name, type_id, rank, dims, maxdims, chunk_dims);
    }
    else
    {
        /* fixed dataset -- determine dimensionality */
        
        if ((xdim > 0) && (ydim > 0) && (zdim > 0))
        {
            rank = 3;
            dims[0] = xdim;
            dims[1] = ydim;
            dims[2] = zdim;
        }
        else if ((xdim > 0) && (ydim > 0))
        {
            rank = 2;
            dims[0] = xdim;
            dims[1] = ydim;
        }
        else if (zdim > 0)
        {
            rank = 1;
            dims[0] = zdim;
        }

        /* if field has components, update last dimension */
        if (cdim > 0)
        {
            rank += 1;
            dims[rank-1] = cdim;
        }
        
        /* finally, create the array */
        h5create_array(loc_id, name, type_id, rank, dims, NULL, NULL);
    }
}

static void h5write_field(hid_t dset_id,
                          hid_t mem_type_id,
                          const void *data,
                          int tdim, int xdim, int ydim, int zdim, int cdim,
                          int nx, int ny, int nz,
                          int px, int py, int pz)
{
    hid_t dataset;
    herr_t status;

    int rank;
    hsize_t size[5]    = {0,0,0,0,0};
    //hsize_t memdims[5] = {0,0,0,0,0};
    hsize_t offset[5]  = {0,0,0,0,0};
    hsize_t stride[5]  = {1,1,1,1,1};
    hsize_t count[5]   = {1,1,1,1,1};
    hsize_t block[5]   = {0,0,0,0,0}; // XXX: always equal to memdims[]?

    dataset = H5Dopen(loc_id, name);
    
    if (tdim > 0)
    {
        if ((xdim > 0) && (ydim > 0) && (zdim > 0))
        {
            rank = 1 + 3;

            size[0] = tdim;
            size[1] = xdim;
            size[2] = ydim;
            size[3] = zdim;

            offset[0] = tdim;
            offset[1] = px*nx;
            offset[2] = py*ny;
            offset[3] = pz*nz;

            block[0] = 1;
            block[1] = nx;
            block[2] = ny;
            block[3] = nz;
        }
        else if ((xdim > 0) && (ydim > 0))
        {
            rank = 1 + 2;

            size[0] = tdim;
            size[1] = xdim;
            size[2] = ydim;

            offset[0] = tdim;
            offset[1] = px*nx;
            offset[2] = py*ny;

            block[0] = 1;
            block[1] = nx;
            block[2] = ny;

        }
        else if (zdim > 0)
        {
            rank = 1 + 1;

            size[0] = tdim;
            size[1] = zdim;

            offset[0] = tdim;
            offset[1] = pz*nz;

            block[0] = 1;
            block[1] = nz;
        }

        if (cdim > 0)
        {
            rank += 1;
            size[rank-1] = cdim;
            offset[rank-1] = 0;
            block[rank-1] = cdim;
        }
        
        h5write_array_hyperslab(dataset, mem_type_id, data,
                                rank, size, block,
                                offset, stride, count, block);
    }
    else
    {

        if ((xdim > 0) && (ydim > 0) && (zdim > 0))
        {
            rank = 3;

            size[0] = xdim;
            size[1] = ydim;
            size[2] = zdim;

            offset[0] = px*nx;
            offset[1] = py*ny;
            offset[2] = pz*nz;

            block[0] = nx;
            block[1] = ny;
            block[2] = nz;
        }
        else if ((xdim > 0) && (ydim > 0))
        {
            rank = 2;

            size[0] = xdim;
            size[1] = ydim;

            offset[0] = px*nx;
            offset[1] = py*ny;

            block[0] = nx;
            block[1] = ny;
        }
        else if (zdim > 0)
        {
            rank = 1;

            size[0] = zdim;
            offset[0] = pz*nz;
            block[0] = nz;
        }

        if (cdim > 0)
        {
            rank += 1;
            size[rank-1] = cdim;
            offset[rank-1] = 0;
            block[rank-1] = cdim;
        }

        h5write_array_hyperslab(dataset, mem_type_id, data,
                                rank, NULL, block,
                                offset, stride, count, block);
    }

    status = H5Dclose(dataset);
}

static void h5create_coord(hid_t loc_id, hid_t type_id, int nodex, int nodey, int nodez)
{
    h5create_field(loc_id, "coord", type_id, 0, nodex, nodey, nodez, 3);
}

static void h5create_velocity(hid_t loc_id, hid_t type_id, int nodex, int nodey, int nodez)
{
    h5create_field(loc_id, "velocity", type_id, 1, nodex, nodey, nodez, 3);
}

static void h5create_temperature(hid_t loc_id, hid_t type_id, int nodex, int nodey, int nodez)
{
    h5create_field(loc_id, "temperature", type_id, 1, nodex, nodey, nodez, 0);
}

static void h5create_viscosity(hid_t loc_id, hid_t type_id, int nodex, int nodey, int nodez)
{
    h5create_field(loc_id, "viscosity", type_id, 1, nodex, nodey, nodez, 0);
}

static void h5create_pressure(hid_t loc_id, hid_t type_id, int nodex, int nodey, int nodez)
{
    h5create_field(loc_id, "pressure", type_id, 1, nodex, nodey, nodez, 0);
}

static void h5create_stress(hid_t loc_id, hid_t type_id, int nodex, int nodey, int nodez)
{
    h5create_field(loc_id, "stress", type_id, 1, nodex, nodey, nodez, 6);
}

static void h5create_surf_coord(hid_t loc_id, hid_t type_id, int nodex, int nodey)
{
    h5create_field(loc_id, "coord", type_id, 0, nodex, nodey, 0, 2);
}

static void h5create_surf_velocity(hid_t loc_id, hid_t type_id, int nodex, int nodey)
{
    h5create_field(loc_id, "velocity", type_id, 1, nodex, nodey, 0, 2);
}

static void h5create_surf_heatflux(hid_t loc_id, hid_t type_id, int nodex, int nodey)
{
    h5create_field(loc_id, "heatflux", type_id, 1, nodex, nodey, 0, 0);
}

static void h5create_surf_topography(hid_t loc_id, hid_t type_id, int nodex, int nodey)
{
    h5create_field(loc_id, "topography", type_id, 1, nodex, nodey, 0, 0);
}



#endif


/****************************************************************************
 * Functions to save specific physical quantities as HDF5 arrays.           *
 ****************************************************************************/

void h5output_coord(struct All_variables *E)
{
#ifdef USE_HDF5

    int cap;
    hid_t cap_group;

    hid_t dataset;
    hid_t filespace;
    hid_t memspace;
    hid_t dxpl_id;

    herr_t status;

    int rank = 4;
    int dims[4];

    hsize_t offset[4] = {0,0,0,0};
    hsize_t stride[4] = {1,1,1,1};
    hsize_t count[4]  = {1,1,1,1};
    hsize_t block[4]  = {0,0,0,0};
    
    /* determine current cap and open group */

    cap = E->hdf5.capid;
    cap_group = H5Gopen(E->hdf5.file_id, E->hdf5.cap_groups[cap]);

    dataset = H5Dopen(cap_group, "coord");
    filespace = H5Dget_space(dataset);

    /* dataset transfer property list -- collective dataset write*/
    dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    status  = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);
    //status  = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_INDEPENDENT);

    /* create memory dataspace */
    rank = 4;
    dims[0] = E->mesh.nox;
    dims[1] = E->mesh.noy;
    dims[2] = E->mesh.noz;
    dims[3] = 3;
    memspace = H5Screate_simple(rank, dims, NULL);

    /* hyperslab selection */
    
    offset[0] = (E->parallel.me_loc[1]) * (E->lmesh.nox);
    offset[1] = (E->parallel.me_loc[2]) * (E->lmesh.noy);
    offset[2] = (E->parallel.me_loc[3]) * (E->lmesh.noz);
    offset[3] = 0;

    block[0] = E->lmesh.nox;
    block[1] = E->lmesh.noy;
    block[2] = E->lmesh.noz;
    block[3] = 3;

    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                 offset, stride, count, block);
    
    /* prepare the data -- citcom yxz order to xyz order */

    E->hdf5.coord = (double *)malloc((E->mesh.nno) * sizeof(double));

    /* write the data to the hyperslab */

    status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, filespace,
                      dxpl_id, &(E->hdf5.coord));
    
    /* release resources */
    free(E->hdf5.coord);
    status = H5Pclose(dxpl_id);
    status = H5Sclose(memspace);
    status = H5Sclose(filespace);
    status = H5Dclose(dataset);
    status = H5Gclose(cap_group);

#endif
}

void h5output_velocity(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_temperature(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_viscosity(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_pressure(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_stress(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_material(struct All_variables *E)
{
#ifdef USE_HDF5

#endif
}

void h5output_tracer(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_surf_botm(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_surf_botm_pseudo_surf(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_avg_r(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

