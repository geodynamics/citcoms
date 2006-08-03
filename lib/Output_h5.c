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
static hid_t h5create_file(const char *filename,
                           unsigned flags,
                           hid_t fcpl_id,
                           hid_t fapl_id);

static hid_t h5create_group(hid_t loc_id,
                            const char *name,
                            size_t size_hint);

static void h5create_dataset(hid_t loc_id,
                             const char *name,
                             hid_t type_id,
                             int rank,
                             hsize_t *dims,
                             hsize_t *maxdims,
                             hsize_t *chunkdims);

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
    void h5output_open(struct  All_variables *E);
    
    printf("h5output()\n");
    if (cycles == 0) {
        h5output_open(E);
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

    if(E->control.tracer == 1)
        h5output_tracer(E, cycles);

    if(E->control.stress == 1)
        h5output_stress(E, cycles);

    if(E->control.pressure == 1)
        h5output_pressure(E, cycles);

    /* disable horizontal average h5output   by Tan2 */
    /* h5output_ave_r(E, cycles); */

    /* prepare for next step */
    E->hdf5.step += 1;

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

    hid_t file_id;      /* HDF5 file identifier */
    hid_t fcpl_id;      /* file creation property list identifier */
    hid_t fapl_id;      /* file access property list identifier */

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
    int nx, ny, nz;

    printf("h5output_open()\n");

    /* determine filename */
    strncpy(E->hdf5.filename, E->control.data_file, (size_t)99);
    strncat(E->hdf5.filename, ".h5", (size_t)99);
    printf("\tfilename = \"%s\"\n", E->hdf5.filename);

    /* set up file creation property list with defaults */
    fcpl_id = H5P_DEFAULT;

    /* set up file access property list with parallel I/O access */
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    status = H5Pset_fapl_mpio(fapl_id, comm, info);

    /* create a new file collectively and release property list identifier */
    file_id = h5create_file(E->hdf5.filename, H5F_ACC_TRUNC, fcpl_id, fapl_id);
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

    /* determine dimensions of local mesh */
    nx = E->lmesh.nox;
    ny = E->lmesh.noy;
    nz = E->lmesh.noz;

    /* type to use for creating the fields */

    E->hdf5.type_id = H5T_NATIVE_FLOAT;

    type_id = E->hdf5.type_id;

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
        //h5create_viscosity(cap_group, type_id, nodex, nodey, nodez);
        //h5create_pressure(cap_group, type_id, nodex, nodey, nodez);
        //h5create_stress(cap_group, type_id, nodex, nodey, nodez);

        /********************************************************************
         * Create /cap/surf/ group                                          *
         ********************************************************************/

        //surf_group = h5create_group(cap_group, "surf", (size_t)0);

        //h5create_surf_coord(surf_group, type_id, nodex, nodey);
        //h5create_surf_velocity(surf_group, type_id, nodex, nodey); 
        //h5create_surf_heatflux(surf_group, type_id, nodex, nodey);
        //h5create_surf_topography(surf_group, type_id, nodex, nodey);
        //status = H5Gclose(surf_group);


        /********************************************************************
         * Create /cap/botm/ group                                          *
         ********************************************************************/

        //botm_group = h5create_group(cap_group, "botm", (size_t)0);

        //h5create_surf_coord(botm_group, type_id, nodex, nodey);
        //h5create_surf_velocity(botm_group, type_id, nodex, nodey); 
        //h5create_surf_heatflux(botm_group, type_id, nodex, nodey);
        //h5create_surf_topography(botm_group, type_id, nodex, nodey);
        //status = H5Gclose(botm_group);

        /* release resources */
        status = H5Gclose(cap_group);
    }
    //status = H5Fclose(file_id);

    /* allocate buffers */
    E->hdf5.vector3d = (double *)malloc((nx*ny*nz*3)*sizeof(double));
    E->hdf5.scalar3d = (double *)malloc((nx*ny*nz)*sizeof(double));
    E->hdf5.vector2d = (double *)malloc((nx*ny*2)*sizeof(double));
    E->hdf5.scalar2d = (double *)malloc((nx*ny)*sizeof(double));
    E->hdf5.scalar1d = (double *)malloc((nz)*sizeof(double));

    /* step about to be executed */
    E->hdf5.step = 0;

#endif
}

void h5output_close(struct All_variables *E)
{
#ifdef USE_HDF5
    herr_t status = H5Fclose(E->hdf5.file_id);
    free(E->hdf5.vector3d);
    free(E->hdf5.scalar3d);
    free(E->hdf5.vector2d);
    free(E->hdf5.scalar2d);
    free(E->hdf5.scalar1d);
#endif
}



/*****************************************************************************
 * Private functions to simplify certain tasks in the h5output_*() functions *
 *****************************************************************************/
#ifdef USE_HDF5

/* Function to create an HDF5 file compatible with PyTables.
 *
 * To enable parallel I/O access, use something like the following:
 *
 * hid_t file_id;
 * hid_t fcpl_id, fapl_id;
 * herr_t status;
 *
 * MPI_Comm comm = MPI_COMM_WORLD;
 * MPI_Info info = MPI_INFO_NULL;
 *
 * ...
 *
 * fcpl_id = H5P_DEFAULT;
 *
 * fapl_id = H5Pcreate(H5P_FILE_ACCESS);
 * status  = H5Pset_fapl_mpio(fapl_id, comm, info);
 *
 * file_id = h5create_file(filename, H5F_ACC_TRUNC, fcpl_id, fapl_id);
 * status  = H5Pclose(fapl_id);
 */
static hid_t h5create_file(const char *filename,
                           unsigned flags,
                           hid_t fcpl_id,
                           hid_t fapl_id)
{
    hid_t file_id;
    hid_t root;

    herr_t status;

    /* Create the HDF5 file */
    file_id = H5Fcreate(filename, flags, fcpl_id, fapl_id);

    /* Write necessary attributes to root group for PyTables compatibility */
    root = H5Gopen(file_id, "/");
    set_attribute(root, "TITLE", "CitcomS output");
    set_attribute(root, "CLASS", "GROUP");
    set_attribute(root, "VERSION", "1.0");
    set_attribute(root, "FILTERS", FILTERS_P);
    set_attribute(root, "PYTABLES_FORMAT_VERSION", "1.5");

    /* release resources */
    status = H5Gclose(root);

    return file_id;
}

static hid_t h5create_group(hid_t loc_id, const char *name, size_t size_hint)
{
    hid_t group_id;
    
    /* TODO:
     *  Make sure this function is called with an appropriately
     *  estimated size_hint parameter
     */
    group_id = H5Gcreate(loc_id, name, size_hint);
    printf("h5create_group()\n");
    printf("\tname=\"%s\"\n", name);

    /* Write necessary attributes for PyTables compatibility */
    set_attribute(group_id, "TITLE", "CitcomS HDF5 group");
    set_attribute(group_id, "CLASS", "GROUP");
    set_attribute(group_id, "VERSION", "1.0");
    set_attribute(group_id, "FILTERS", FILTERS_P);
    set_attribute(group_id, "PYTABLES_FORMAT_VERSION", "1.5");
    
    return group_id;
}

static hid_t h5open_cap(struct All_variables *E)
{
    hid_t cap_group = H5Gopen(E->hdf5.file_id,
                              E->hdf5.cap_groups[E->hdf5.capid]);
    ///* DEBUG
    printf("\th5open_cap()\n");
    printf("\t\tOpening capid %d: %s\n", E->hdf5.capid,
           E->hdf5.cap_groups[E->hdf5.capid]);
    // */
    return cap_group;
}

static void h5create_dataset(hid_t loc_id,
                             const char *name,
                             hid_t type_id,
                             int rank,
                             hsize_t *dims,
                             hsize_t *maxdims,
                             hsize_t *chunkdims)
{
    hid_t dcpl_id;      /* dataset creation property list identifier */
    hid_t dataspace;    /* file dataspace identifier */
    hid_t dataset;      /* dataset identifier */
    herr_t status;

    ///* DEBUG
    printf("\t\th5create_dataset()\n");
    printf("\t\t\tname=\"%s\"\n", name);
    printf("\t\t\trank=%d\n", rank);
    printf("\t\t\tdims={%d,%d,%d,%d,%d}\n",
        (int)dims[0], (int)dims[1], (int)dims[2], (int)dims[3], (int)dims[4]);
    if(maxdims != NULL) 
        printf("\t\t\tmaxdims={%d,%d,%d,%d,%d}\n",
            (int)maxdims[0], (int)maxdims[1], (int)maxdims[2],
            (int)maxdims[3], (int)maxdims[4]);
    else
        printf("\t\t\tmaxdims=NULL\n");
    if(chunkdims != NULL)
        printf("\t\t\tchunkdims={%d,%d,%d,%d,%d}\n",
            (int)chunkdims[0], (int)chunkdims[1], (int)chunkdims[2],
            (int)chunkdims[3], (int)chunkdims[4]);
    else
        printf("\t\t\tchunkdims=NULL\n");
    // */

    /* create the dataspace for the dataset */
    dataspace = H5Screate_simple(rank, dims, maxdims);
    
    dcpl_id = H5P_DEFAULT;
    if (chunkdims != NULL)
    {
        /* modify dataset creation properties to enable chunking */
        dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
        status = H5Pset_chunk(dcpl_id, rank, chunkdims);
        //status = H5Pset_fill_value(dcpl_id, H5T_NATIVE_FLOAT, &fillvalue);
    }

    /* create the dataset */
    dataset = H5Dcreate(loc_id, name, type_id, dataspace, dcpl_id);

    /* Write necessary attributes for PyTables compatibility */
    set_attribute(dataset, "TITLE", "CitcomS HDF5 dataset"); //TODO: elsewhere?
    set_attribute(dataset, "CLASS", "ARRAY");
    set_attribute(dataset, "FLAVOR", "numpy");
    set_attribute(dataset, "VERSION", "2.3");

    /* release resources */
    if(chunkdims != NULL)
    {
        status = H5Pclose(dcpl_id);
    }
    status = H5Sclose(dataspace);
    status = H5Dclose(dataset);
    return;
}

/* Function for creating a Citcom field
 *
 *  tdim - extent of temporal dimension
 *          if tdim >= 0, field will not contain a temporal dimension
 *          if tdim < 0, field will have time as its first dimension
 *
 *  xdim - spatial extent along x-direction
 *          if xdim <= 0, field will be one dimensional (along z-direction)
 *
 *  ydim - spatial extent along y-direction
 *          if ydim <= 0, field will be one dimensional (along z-direction)
 *
 *  zdim - spatial extent along z-direction
 *          if zdim <= 0, field will be two dimensional (along xy-plane)
 *
 *  cdim - dimensions of a single point in the field
 *          if cdim=0, we have a scalar field (i.e., 0 components)
 *          if cdim=2, we have a vector field with 2 components (xy plane)
 *          if cdim=3, we have a vector field with 3 components (xyz space)
 *          if cdim=6, we have a symmetric tensor field with 6 components
 */
static void h5create_field(hid_t loc_id,
                           const char *name,
                           hid_t type_id,
                           int tdim,
                           int xdim,
                           int ydim,
                           int zdim,
                           int cdim)
{
    int nsd = 0;
    int rank = 0;
    hsize_t dims[5]      = {0,0,0,0,0};
    hsize_t maxdims[5]   = {0,0,0,0,0};
    hsize_t chunkdims[5] = {0,0,0,0,0};
    herr_t status;

    int t = -100;
    int x = -100;
    int y = -100;
    int z = -100;
    int c = -100;

    /* Assign default indices according to dimensionality
     *
     *  nsd - number of spatial dimensions
     *          if nsd=3, field is a three-dimensional volume (xyz)
     *          if nsd=2, field is a two-dimensional surface (xy)
     *          if nsd=1, field is a one-dimensional set (along z)
     */
    if ((xdim > 0) && (ydim > 0) && (zdim > 0))
    {
        nsd = 3;
        x = 0;
        y = 1;
        z = 2;
    }
    else if ((xdim > 0) && (ydim > 0))
    {
        nsd = 2;
        x = 0;
        y = 1;
    }
    else if (zdim > 0)
    {
        nsd = 1;
        z = 0;
    }

    /* Rank increases by the number of spatial dimensions found */
    rank += nsd;

    /* Rank increases by one for time-varying datasets. Note that
     * since temporal dimension goes first, the spatial indices are
     * shifted up by one (which explains why the default value for
     * the positional indices x,y,z is not just -1).
     */
    if (tdim >= 0)
    {
        rank += 1;
        t  = 0;
        x += 1;
        y += 1;
        z += 1;
    }

    /* Rank increases by one if components are present. Note that
     * the dimension used for the components is the last dimension.
     */
    if (cdim > 0)
    {
        rank += 1;
        c = rank-1;
    }


    /* Finally, construct the appropriate dataspace parameters */
    if (nsd > 0)
    {
        if (t >= 0)
        {
            dims[t] = tdim;
            maxdims[t] = H5S_UNLIMITED;
            chunkdims[t] = 1;
        }
        
        if (x >= 0)
        {
            dims[x] = xdim;
            maxdims[x] = xdim;
            chunkdims[x] = xdim;
        }

        if (y >= 0)
        {
            dims[y] = ydim;
            maxdims[y] = ydim;
            chunkdims[y] = ydim;
        }

        if (z >= 0)
        {
            dims[z] = zdim;
            maxdims[z] = zdim;
            chunkdims[z] = zdim;
        }

        if (c >= 0)
        {
            dims[c] = cdim;
            maxdims[c] = cdim;
            chunkdims[c] = cdim;
        }

    }

    ///* DEBUG
    printf("\th5create_field()\n");
    printf("\t\tname=\"%s\"\n", name);
    printf("\t\tshape=(%d,%d,%d,%d,%d)\n",
           tdim, xdim, ydim, zdim, cdim);
    printf("\t\trank=%d\n", rank);
    printf("\t\tdims={%d,%d,%d,%d,%d}\n",
            (int)dims[0], (int)dims[1], (int)dims[2],
            (int)dims[3], (int)dims[4]);
    printf("\t\tmaxdims={%d,%d,%d,%d,%d}\n",
            (int)maxdims[0], (int)maxdims[1], (int)maxdims[2],
            (int)maxdims[3], (int)maxdims[4]);
    if (tdim >= 0)
        printf("\t\tchunkdims={%d,%d,%d,%d,%d}\n",
            (int)chunkdims[0], (int)chunkdims[1], (int)chunkdims[2],
            (int)chunkdims[3], (int)chunkdims[4]);
    else
        printf("\t\tchunkdims=NULL\n");
    // */


    if (tdim >= 0)
        h5create_dataset(loc_id, name, type_id,
                         rank, dims, maxdims, chunkdims);
    else
        h5create_dataset(loc_id, name, type_id,
                         rank, dims, maxdims, NULL);
}

static void h5write_dataset(hid_t dset_id,
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
    hid_t filespace;    /* file dataspace */
    hid_t memspace;     /* memory dataspace */
    hid_t dxpl_id;      /* dataset transfer property list identifier */
    herr_t status;
    
    ///* DEBUG
    printf("\th5write_dataset()\n");
    printf("\t\trank    = %d\n", rank);
    if(size != NULL)
        printf("\t\tsize    = {%d,%d,%d,%d,%d}\n",
            (int)size[0], (int)size[1], (int)size[2],
            (int)size[3], (int)size[4]);
    else
        printf("\t\tsize    = NULL\n");
    printf("\t\tmemdims = {%d,%d,%d,%d,%d}\n",
        (int)memdims[0], (int)memdims[1], (int)memdims[2],
        (int)memdims[3], (int)memdims[4]);
    printf("\t\toffset  = {%d,%d,%d,%d,%d}\n",
        (int)offset[0], (int)offset[1], (int)offset[2],
        (int)offset[3], (int)offset[4]);
    printf("\t\tstride  = {%d,%d,%d,%d,%d}\n",
        (int)stride[0], (int)stride[1], (int)stride[2],
        (int)stride[3], (int)stride[4]);
    printf("\t\tcount   = {%d,%d,%d,%d,%d}\n",
        (int)count[0], (int)count[1], (int)count[2],
        (int)count[3], (int)count[4]);
    printf("\t\tblock   = {%d,%d,%d,%d,%d}\n",
        (int)block[0], (int)block[1], (int)block[2],
        (int)block[3], (int)block[4]);
    // */

    /* extend the dataset if necessary */
    if(size != NULL)
    {
        printf("\t\tExtending dataset\n");
        status = H5Dextend(dset_id, size);
    }

    /* get file dataspace */
    printf("\t\tGetting file dataspace from dataset\n");
    filespace = H5Dget_space(dset_id);

    /* create memory dataspace */
    printf("\t\tCreating memory dataspace\n");
    memspace = H5Screate_simple(rank, memdims, NULL);

    /* hyperslab selection */
    printf("\t\tSelecting hyperslab in file dataspace\n");
    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                 offset, stride, count, block);

    /* dataset transfer property list */
    printf("\t\tSetting dataset transfer to H5FD_MPIO_COLLECTIVE\n");
    dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    status  = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);
    //status = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_INDEPENDENT);

    /* write the data to the hyperslab */
    printf("\t\tWriting data to the hyperslab\n");
    status = H5Dwrite(dset_id, mem_type_id, memspace, filespace, dxpl_id, data);
    
    /* release resources */
    status = H5Pclose(dxpl_id);
    status = H5Sclose(memspace);
    status = H5Sclose(filespace);
    return;
}


static void h5write_field(hid_t dset_id,
                          hid_t mem_type_id,
                          const void *data,
                          int tdim, int xdim, int ydim, int zdim, int cdim,
                          struct All_variables *E)
{
    int nsd = 0;
    int step = 0;

    int rank = 0;
    hsize_t size[5]    = {0,0,0,0,0};
    hsize_t offset[5]  = {0,0,0,0,0};
    hsize_t stride[5]  = {1,1,1,1,1};
    hsize_t count[5]   = {1,1,1,1,1};
    hsize_t block[5]   = {0,0,0,0,0};

    int t = -100;
    int x = -100;
    int y = -100;
    int z = -100;
    int c = -100;

    int nx = E->lmesh.nox;
    int ny = E->lmesh.noy;
    int nz = E->lmesh.noz;

    int nprocx = E->parallel.nprocx;
    int nprocy = E->parallel.nprocy;
    int nprocz = E->parallel.nprocz;

    int px = E->parallel.me_loc[1];
    int py = E->parallel.me_loc[2];
    int pz = E->parallel.me_loc[3];


    /* Refer to h5create_field() for more detailed comments. */

    if ((xdim > 0) && (ydim > 0) && (zdim > 0))
    {
        nsd = 3;
        x = 0;
        y = 1;
        z = 2;
    }
    else if ((xdim > 0) && (ydim > 0))
    {
        nsd = 2;
        x = 0;
        y = 1;
    }
    else if (zdim > 0)
    {
        nsd = 1;
        z = 0;
    }

    rank += nsd;

    if (tdim > 0)
    {
        rank += 1;
        t  = 0;
        x += 1;
        y += 1;
        z += 1;
    }
    
    if (cdim > 0)
    {
        rank += 1;
        c = rank-1;
    }

    if (nsd > 0)
    {
        if (t >= 0)
        {
            size[t]   = tdim;
            offset[t] = tdim-1;
            block[t]  = 1;
        }

        if (x >= 0)
        {
            size[x]   = xdim;
            offset[x] = px*(nx-1);
            block[x]  = (px == nprocx-1) ? nx : nx-1;
        }

        if (y >= 0)
        {
            size[y]   = ydim;
            offset[y] = py*(ny-1);
            block[y]  = (py == nprocy-1) ? ny : ny-1;
        }

        if (z >= 0)
        {
            size[z]   = zdim;
            offset[z] = pz*(nz-1);
            block[z]  = (pz == nprocz-1) ? nz : nz-1;
        }

        if (c >= 0)
        {
            size[c]   = cdim;
            offset[c] = 0;
            block[c]  = cdim;
        }

        if (tdim > 0)
            h5write_dataset(dset_id, mem_type_id, data,
                            rank, size, block,
                            offset, stride, count, block);
        else
            h5write_dataset(dset_id, mem_type_id, data,
                            rank, NULL, block,
                            offset, stride, count, block);
    }

    return;
}

static void h5create_coord(hid_t loc_id, hid_t type_id, int nodex, int nodey, int nodez)
{
    h5create_field(loc_id, "coord", type_id, -1, nodex, nodey, nodez, 3);
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
    h5create_field(loc_id, "coord", type_id, -1, nodex, nodey, 0, 2);
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

    hid_t cap_group;
    hid_t dataset;
    herr_t status;

    int i, j, k;
    int n, nx, ny, nz;
    int m, mx, my, mz;
    int p, px, py, pz;

    int nodex = E->mesh.nox;
    int nodey = E->mesh.noy;
    int nodez = E->mesh.noz;

    int nprocx = E->parallel.nprocx;
    int nprocy = E->parallel.nprocy;
    int nprocz = E->parallel.nprocz;

    p  = E->parallel.me;
    px = E->parallel.me_loc[1];
    py = E->parallel.me_loc[2];
    pz = E->parallel.me_loc[3];

    nx = E->lmesh.nox;
    ny = E->lmesh.noy;
    nz = E->lmesh.noz;

    mx = (p == nprocx-1) ? nx : nx-1;
    my = (p == nprocy-1) ? ny : ny-1;
    mz = (p == nprocz-1) ? nz : nz-1;
    
    /* prepare the data -- change citcom yxz order to xyz order */
    for(i = 0; i < mx; i++)
    {
        for(j = 0; j < my; j++)
        {
            for(k = 0; k < mz; k++)
            {
                n = k + i*nz + j*nz*nx;
                m = k + j*mz + i*mz*my;
                E->hdf5.vector3d[3*m+0] = E->sx[1][1][n+1];
                E->hdf5.vector3d[3*m+1] = E->sx[1][2][n+1];
                E->hdf5.vector3d[3*m+2] = E->sx[1][3][n+1];
            }
        }
    }

    printf("h5output_coord()\n");

    /* write to dataset */
    cap_group = h5open_cap(E);
    dataset = H5Dopen(cap_group, "coord");
    h5write_field(dataset, H5T_NATIVE_DOUBLE, E->hdf5.vector3d,
                  -1, nodex, nodey, nodez, 3, E);

    /* release resources */
    status = H5Dclose(dataset);
    status = H5Gclose(cap_group);

#endif
}

void h5output_velocity(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5
    hid_t cap;
    hid_t dataset;
    herr_t status;

    int i, j, k;
    int n, nx, ny, nz;
    int m, mx, my, mz;
    int p, px, py, pz;

    int nodex = E->mesh.nox;
    int nodey = E->mesh.noy;
    int nodez = E->mesh.noz;

    int nprocx = E->parallel.nprocx;
    int nprocy = E->parallel.nprocy;
    int nprocz = E->parallel.nprocz;

    p  = E->parallel.me;
    px = E->parallel.me_loc[1];
    py = E->parallel.me_loc[2];
    pz = E->parallel.me_loc[3];

    nx = E->lmesh.nox;
    ny = E->lmesh.noy;
    nz = E->lmesh.noz;

    mx = (p == nprocx-1) ? nx : nx-1;
    my = (p == nprocy-1) ? ny : ny-1;
    mz = (p == nprocz-1) ? nz : nz-1;
    
    /* prepare the data -- change citcom yxz order to xyz order */
    for(i = 0; i < mx; i++)
    {
        for(j = 0; j < my; j++)
        {
            for(k = 0; k < mz; k++)
            {
                n = k + i*nz + j*nz*nx;
                m = k + j*mz + i*mz*my;
                E->hdf5.vector3d[3*m+0] = E->sphere.cap[1].V[1][n+1];
                E->hdf5.vector3d[3*m+1] = E->sphere.cap[1].V[2][n+1];
                E->hdf5.vector3d[3*m+2] = E->sphere.cap[1].V[3][n+1];
            }
        }
    }

    printf("h5output_velocity()\n");

    cap = h5open_cap(E);
    dataset = H5Dopen(cap, "velocity");
    h5write_field(dataset, H5T_NATIVE_DOUBLE, E->hdf5.vector3d,
                  E->hdf5.step+1, nodex, nodey, nodez, 3, E);

    /* release resources */
    status = H5Dclose(dataset);
    status = H5Gclose(cap);
#endif
}

void h5output_temperature(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5
    hid_t cap;
    hid_t dataset;
    herr_t status;

    int i, j, k;
    int n, nx, ny, nz;
    int m, mx, my, mz;
    int p, px, py, pz;

    int nodex = E->mesh.nox;
    int nodey = E->mesh.noy;
    int nodez = E->mesh.noz;

    int nprocx = E->parallel.nprocx;
    int nprocy = E->parallel.nprocy;
    int nprocz = E->parallel.nprocz;

    p  = E->parallel.me;
    px = E->parallel.me_loc[1];
    py = E->parallel.me_loc[2];
    pz = E->parallel.me_loc[3];

    nx = E->lmesh.nox;
    ny = E->lmesh.noy;
    nz = E->lmesh.noz;

    mx = (p == nprocx-1) ? nx : nx-1;
    my = (p == nprocy-1) ? ny : ny-1;
    mz = (p == nprocz-1) ? nz : nz-1;
    
    /* prepare the data -- change citcom yxz order to xyz order */
    for(i = 0; i < mx; i++)
    {
        for(j = 0; j < my; j++)
        {
            for(k = 0; k < mz; k++)
            {
                n = k + i*nz + j*nz*nx;
                m = k + j*mz + i*mz*my;
                E->hdf5.scalar3d[m] = E->T[1][n+1];
            }
        }
    }

    printf("h5output_temperature()\n");

    cap = h5open_cap(E);
    dataset = H5Dopen(cap, "temperature");
    h5write_field(dataset, H5T_NATIVE_DOUBLE, E->hdf5.scalar3d,
                  E->hdf5.step+1, nodex, nodey, nodez, 0, E);

    /* release resources */
    status = H5Dclose(dataset);
    status = H5Gclose(cap);
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

