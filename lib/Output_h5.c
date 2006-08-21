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

/* Routines to write the output of the finite element cycles
 * into an HDF5 file, using parallel I/O.
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

#ifdef USE_HDF5

/* constant data (only for first cycle) */
void h5output_meta(struct All_variables *);
void h5output_coord(struct All_variables *);
void h5output_material(struct All_variables *);

/* time-varying data */
void h5extend_time_dimension(struct All_variables *E);
void h5output_velocity(struct All_variables *, int);
void h5output_temperature(struct All_variables *, int);
void h5output_viscosity(struct All_variables *, int);
void h5output_pressure(struct All_variables *, int);
void h5output_stress(struct All_variables *, int);
void h5output_tracer(struct All_variables *, int);
void h5output_surf_botm(struct All_variables *, int);
void h5output_average(struct All_variables *, int);
void h5output_time(struct All_variables *, int);

/* for creation of HDF5 objects (wrapped for PyTables compatibility) */
static hid_t h5create_file(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id);
static hid_t h5create_group(hid_t loc_id, const char *name, size_t size_hint);
static herr_t h5create_dataset(hid_t loc_id, const char *name, const char *title, hid_t type_id, int rank, hsize_t *dims, hsize_t *maxdims, hsize_t *chunkdims);

/* for creation of field and dataset objects */
static herr_t h5allocate_field(struct All_variables *E, enum field_class_t field_class, int nsd, int time, hid_t dtype, field_t **field);
static herr_t h5create_field(hid_t loc_id, const char *name, const char *title, field_t *field);
static herr_t h5create_coord(hid_t loc_id, field_t *field);
static herr_t h5create_velocity(hid_t loc_id, field_t *field);
static herr_t h5create_stress(hid_t loc_id, field_t *field);
static herr_t h5create_temperature(hid_t loc_id, field_t *field);
static herr_t h5create_viscosity(hid_t loc_id, field_t *field);
static herr_t h5create_pressure(hid_t loc_id, field_t *field);
static herr_t h5create_surf_coord(hid_t loc_id, field_t *field);
static herr_t h5create_surf_velocity(hid_t loc_id, field_t *field);
static herr_t h5create_surf_heatflux(hid_t loc_id, field_t *field);
static herr_t h5create_surf_topography(hid_t loc_id, field_t *field);
static herr_t h5create_time(hid_t loc_id);

/* for writing to datasets */
static herr_t h5write_dataset(hid_t dset_id, hid_t mem_type_id, const void *data, int rank, hsize_t *memdims, hsize_t *offset, hsize_t *stride, hsize_t *count, hsize_t *block);
static herr_t h5write_field(hid_t dset_id, field_t *field);

/* for releasing resources */
static herr_t h5close_field(field_t **field);


#endif

extern void parallel_process_termination();
extern void heat_flux(struct All_variables *);
extern void get_STD_topo(struct All_variables *, float**, float**, float**, float**, int);


/****************************************************************************
 * Functions that control which data is saved to output file(s).            *
 * These represent possible choices for (E->output) function pointer.       *
 ****************************************************************************/

void h5output(struct All_variables *E, int cycles)
{
#ifndef USE_HDF5
    if(E->parallel.me == 0)
        fprintf(stderr, "h5output(): CitcomS was compiled without HDF5!\n");
    MPI_Finalize();
    exit(8);
#else
    if (cycles == 0) {
        h5output_open(E);
        h5output_meta(E);
        h5output_coord(E);
        h5output_material(E);
    }

    h5extend_time_dimension(E);
    h5output_velocity(E, cycles);
    h5output_temperature(E, cycles);
    h5output_viscosity(E, cycles);

    h5output_surf_botm(E, cycles);

    /* output tracer location if using tracer */
    if(E->control.tracer == 1)
        h5output_tracer(E, cycles);

    /* optiotnal output below */
    if(E->output.stress == 1)
        h5output_stress(E, cycles);

    if(E->output.pressure == 1)
        h5output_pressure(E, cycles);

    if (E->output.average == 1)
	h5output_average(E, cycles);

    /* Call this last (for timing information) */
    h5output_time(E, cycles);

#endif
}


/****************************************************************************
 * Functions to initialize and finalize access to HDF5 output file.         *
 * Responsible for creating all necessary groups, attributes, and arrays.   *
 ****************************************************************************/

/* This function should open the HDF5 file, and create any necessary groups,
 * arrays, and attributes. It should also initialize the hyperslab parameters
 * that will be used for future dataset writes (c.f. field_t objects).
 */
void h5output_open(struct All_variables *E)
{
#ifdef USE_HDF5

    /*
     * Citcom variables
     */

    int cap;
    int caps = E->sphere.caps;
    int nprocx, nprocy, nprocz;

    /*
     * MPI variables
     */

    MPI_Comm comm = E->parallel.world;
    MPI_Info info = MPI_INFO_NULL;

    /*
     * HDF5 variables
     */

    hid_t file_id;      /* HDF5 file identifier */
    hid_t fcpl_id;      /* file creation property list identifier */
    hid_t fapl_id;      /* file access property list identifier */

    char *cap_name;
    hid_t cap_group;    /* group identifier for a given cap */
    hid_t surf_group;   /* group identifier for top cap surface */
    hid_t botm_group;   /* group identifier for bottom cap surface */

    hid_t dtype;        /* datatype for dataset creation */

    herr_t status;

    /*
     * Create HDF5 file using parallel I/O
     */

    /* determine filename */
    strncpy(E->hdf5.filename, E->control.data_file, (size_t)99);
    strncat(E->hdf5.filename, ".h5", (size_t)99);

    /* set up file creation property list with defaults */
    fcpl_id = H5P_DEFAULT;

    /* set up file access property list with parallel I/O access */
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    status  = H5Pset_fapl_mpio(fapl_id, comm, info);

    /* create a new file collectively and release property list identifier */
    file_id = h5create_file(E->hdf5.filename, H5F_ACC_TRUNC, fcpl_id, fapl_id);
    status  = H5Pclose(fapl_id);

    /* save the file identifier for later use */
    E->hdf5.file_id = file_id;

    /*
     * Allocate field objects for use in dataset writes...
     */

    E->hdf5.const_vector3d = NULL;
    E->hdf5.const_vector2d = NULL;
    E->hdf5.tensor3d = NULL;
    E->hdf5.vector3d = NULL;
    E->hdf5.vector2d = NULL;
    E->hdf5.scalar3d = NULL;
    E->hdf5.scalar2d = NULL;
    E->hdf5.scalar1d = NULL;

    dtype = H5T_NATIVE_FLOAT; /* store solutions as floats in .h5 file */

    h5allocate_field(E, VECTOR_FIELD, 3, 0, dtype, &(E->hdf5.const_vector3d));
    h5allocate_field(E, VECTOR_FIELD, 2, 0, dtype, &(E->hdf5.const_vector2d));

    h5allocate_field(E, TENSOR_FIELD, 3, 1, dtype, &(E->hdf5.tensor3d));
    h5allocate_field(E, VECTOR_FIELD, 3, 1, dtype, &(E->hdf5.vector3d));
    h5allocate_field(E, VECTOR_FIELD, 2, 1, dtype, &(E->hdf5.vector2d));

    h5allocate_field(E, SCALAR_FIELD, 3, 1, dtype, &(E->hdf5.scalar3d));
    h5allocate_field(E, SCALAR_FIELD, 2, 1, dtype, &(E->hdf5.scalar2d));
    h5allocate_field(E, SCALAR_FIELD, 1, 1, dtype, &(E->hdf5.scalar1d));

    /*
     * Create time table (to store nondimensional and cpu times)
     */

    h5create_time(file_id);

    /*
     * Create necessary groups and arrays
     */
    for(cap = 0; cap < caps; cap++)
    {
        cap_name = E->hdf5.cap_names[cap];
        snprintf(cap_name, (size_t)7, "/cap%02d", cap);

        /********************************************************************
         * Create /cap/ group                                               *
         ********************************************************************/

        cap_group = h5create_group(file_id, cap_name, (size_t)0);

        h5create_coord(cap_group, E->hdf5.const_vector3d);
        h5create_velocity(cap_group, E->hdf5.vector3d);
        h5create_temperature(cap_group, E->hdf5.scalar3d);
        h5create_viscosity(cap_group, E->hdf5.scalar3d);

	if(E->output.pressure == 1)
	    h5create_pressure(cap_group, E->hdf5.scalar3d);

	if(E->output.stress == 1)
	    h5create_stress(cap_group, E->hdf5.tensor3d);

        /********************************************************************
         * Create /cap/surf/ group                                          *
         ********************************************************************/
	if(E->output.surf == 1) {
        //surf_group = h5create_group(cap_group, "surf", (size_t)0);

        //h5create_surf_coord(surf_group, E->hdf5.const_vector2d);
        //h5create_surf_velocity(surf_group, E->hdf5.vector2d);
        //h5create_surf_heatflux(surf_group, E->hdf5.scalar2d);
        //h5create_surf_topography(surf_group, E->hdf5.scalar2d);
	}

        /********************************************************************
         * Create /cap/botm/ group                                          *
         ********************************************************************/
	if(E->output.botm == 1) {
        //botm_group = h5create_group(cap_group, "botm", (size_t)0);

        //h5create_surf_coord(botm_group, E->hdf5.const_vector2d);
        //h5create_surf_velocity(botm_group, E->hdf5.vector2d);
        //h5create_surf_heatflux(botm_group, E->hdf5.scalar2d);
        //h5create_surf_topography(botm_group, E->hdf5.scalar2d);
        //status = H5Gclose(botm_group);
	}

	if(E->output.average == 1) {
	}

        /* save references to caps */
        E->hdf5.cap_groups[cap] = cap_group;
        //E->hdf5.cap_surf_groups[cap] = surf_group;
        //E->hdf5.cap_botm_groups[cap] = botm_group;

    }
    //status = H5Fclose(file_id);


    /* Number of times we have called h5output() */

    E->hdf5.count = 0; // TODO: for restart, initialize to last value


    /* Determine current cap and remember it */

    nprocx = E->parallel.nprocx;
    nprocy = E->parallel.nprocy;
    nprocz = E->parallel.nprocz;

    cap = (E->parallel.me) / (nprocx * nprocy * nprocz);
    E->hdf5.capid = cap;
    E->hdf5.cap_group = E->hdf5.cap_groups[cap];

#endif
}

/* Finalizing access to HDF5 objects. Note that this function
 * needs to be visible to external files (even when HDF5 isn't
 * configured).
 */
void h5output_close(struct All_variables *E)
{
#ifdef USE_HDF5
    int i;
    herr_t status;

    /* close cap groups */
    for (i = 0; i < E->sphere.caps; i++)
    {
        status = H5Gclose(E->hdf5.cap_groups[i]);
	if(E->output.surf == 1) {
	    //status = H5Gclose(E->hdf5.cap_surf_groups[i]);
	}
	if(E->output.botm == 1) {
	    //status = H5Gclose(E->hdf5.cap_botm_groups[i]);
	}
	if(E->output.average == 1) {
	}
    }

    /* close file */
    status = H5Fclose(E->hdf5.file_id);

    /* close fields (deallocate buffers) */
    h5close_field(&(E->hdf5.const_vector3d));
    h5close_field(&(E->hdf5.const_vector2d));
    h5close_field(&(E->hdf5.vector3d));
    h5close_field(&(E->hdf5.vector2d));
    h5close_field(&(E->hdf5.scalar3d));
    h5close_field(&(E->hdf5.scalar2d));
    h5close_field(&(E->hdf5.scalar1d));
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
    set_attribute_string(root, "TITLE", "CitcomS output");
    set_attribute_string(root, "CLASS", "GROUP");
    set_attribute_string(root, "VERSION", "1.0");
    set_attribute_string(root, "FILTERS", FILTERS_P);
    set_attribute_string(root, "PYTABLES_FORMAT_VERSION", "1.5");

    /* release resources */
    status = H5Gclose(root);

    return file_id;
}


/* Function to create an HDF5 group compatible with PyTables.
 * To close group, call H5Gclose().
 */
static hid_t h5create_group(hid_t loc_id, const char *name, size_t size_hint)
{
    hid_t group_id;

    /* TODO:
     *  Make sure this function is called with an appropriately
     *  estimated size_hint parameter
     */
    group_id = H5Gcreate(loc_id, name, size_hint);

    /* Write necessary attributes for PyTables compatibility */
    set_attribute_string(group_id, "TITLE", "CitcomS HDF5 group");
    set_attribute_string(group_id, "CLASS", "GROUP");
    set_attribute_string(group_id, "VERSION", "1.0");
    set_attribute_string(group_id, "FILTERS", FILTERS_P);
    set_attribute_string(group_id, "PYTABLES_FORMAT_VERSION", "1.5");

    return group_id;
}


static herr_t h5create_dataset(hid_t loc_id,
                               const char *name,
                               const char *title,
                               hid_t type_id,
                               int rank,
                               hsize_t *dims,
                               hsize_t *maxdims,
                               hsize_t *chunkdims)
{
    hid_t dataset;      /* dataset identifier */
    hid_t dataspace;    /* file dataspace identifier */
    hid_t dcpl_id;      /* dataset creation property list identifier */
    herr_t status;

    /* DEBUG
    if (chunkdims != NULL)
        printf("\t\th5create_dataset()\n"
               "\t\t\tname=\"%s\"\n"
               "\t\t\trank=%d\n"
               "\t\t\tdims={%d,%d,%d,%d,%d}\n"
               "\t\t\tmaxdims={%d,%d,%d,%d,%d}\n"
               "\t\t\tchunkdims={%d,%d,%d,%d,%d}\n",
               name, rank,
               (int)dims[0], (int)dims[1],
               (int)dims[2], (int)dims[3], (int)dims[4],
               (int)maxdims[0], (int)maxdims[1],
               (int)maxdims[2], (int)maxdims[3], (int)maxdims[4],
               (int)chunkdims[0], (int)chunkdims[1],
               (int)chunkdims[2],(int)chunkdims[3], (int)chunkdims[4]);
    else if (maxdims != NULL)
        printf("\t\th5create_dataset()\n"
               "\t\t\tname=\"%s\"\n"
               "\t\t\trank=%d\n"
               "\t\t\tdims={%d,%d,%d,%d,%d}\n"
               "\t\t\tmaxdims={%d,%d,%d,%d,%d}\n"
               "\t\t\tchunkdims=NULL\n",
               name, rank,
               (int)dims[0], (int)dims[1],
               (int)dims[2], (int)dims[3], (int)dims[4],
               (int)maxdims[0], (int)maxdims[1],
               (int)maxdims[2], (int)maxdims[3], (int)maxdims[4]);
    else
        printf("\t\th5create_dataset()\n"
               "\t\t\tname=\"%s\"\n"
               "\t\t\trank=%d\n"
               "\t\t\tdims={%d,%d,%d,%d,%d}\n"
               "\t\t\tmaxdims=NULL\n"
               "\t\t\tchunkdims=NULL\n",
               name, rank,
               (int)dims[0], (int)dims[1],
               (int)dims[2], (int)dims[3], (int)dims[4])
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
    set_attribute_string(dataset, "TITLE", title);
    set_attribute_string(dataset, "CLASS", "ARRAY");
    set_attribute_string(dataset, "FLAVOR", "numpy");
    set_attribute_string(dataset, "VERSION", "2.3");

    /* release resources */
    if (chunkdims != NULL)
    {
        status = H5Pclose(dcpl_id);
    }
    status = H5Sclose(dataspace);
    status = H5Dclose(dataset);

    return 0;
}

static herr_t h5allocate_field(struct All_variables *E,
                               enum field_class_t field_class,
                               int nsd,
                               int time,
                               hid_t dtype,
                               field_t **field)
{
    int rank = 0;
    int tdim = 0;
    int cdim = 0;

    /* indices */
    int t = -100;
    int x = -100;
    int y = -100;
    int z = -100;
    int c = -100;

    int dim;

    int px, py, pz;
    int nprocx, nprocy, nprocz;

    int nx, ny, nz;
    int nodex, nodey, nodez;

    /* coordinates of current process in cap */
    px = E->parallel.me_loc[1];
    py = E->parallel.me_loc[2];
    pz = E->parallel.me_loc[3];

    /* dimensions of processes per cap */
    nprocx = E->parallel.nprocx;
    nprocy = E->parallel.nprocy;
    nprocz = E->parallel.nprocz;

    /* determine dimensions of mesh */
    nodex = E->mesh.nox;
    nodey = E->mesh.noy;
    nodez = E->mesh.noz;

    /* determine dimensions of local mesh */
    nx = E->lmesh.nox;
    ny = E->lmesh.noy;
    nz = E->lmesh.noz;

    *field = NULL;

    switch (nsd)
    {
        case 3:
            rank = 3;
            x = 0;
            y = 1;
            z = 2;
            break;
        case 2:
            rank = 2;
            x = 0;
            y = 1;
            break;
        case 1:
            rank = 1;
            z = 0;
            break;
        default:
            return -1;
    }

    if (time > 0)
    {
        rank += 1;
        t  = 0;
        x += 1;
        y += 1;
        z += 1;
    }

    switch (field_class)
    {
        case TENSOR_FIELD:
            cdim = 6;
            rank += 1;
            c = rank-1;
            break;
        case VECTOR_FIELD:
            cdim = nsd;
            rank += 1;
            c = rank-1;
            break;
        case SCALAR_FIELD:
            cdim = 0;
            break;
    }

    if (rank > 0)
    {
        *field = (field_t *)malloc(sizeof(field_t));

        (*field)->dtype = dtype;

        (*field)->rank = rank;
        (*field)->dims = (hsize_t *)malloc(rank * sizeof(hsize_t));
        (*field)->maxdims = (hsize_t *)malloc(rank * sizeof(hsize_t));
        (*field)->chunkdims = NULL;
        if (t >= 0)
            (*field)->chunkdims = (hsize_t *)malloc(rank * sizeof(hsize_t));

        (*field)->offset = (hsize_t *)malloc(rank * sizeof(hsize_t));
        (*field)->stride = (hsize_t *)malloc(rank * sizeof(hsize_t));
        (*field)->count  = (hsize_t *)malloc(rank * sizeof(hsize_t));
        (*field)->block  = (hsize_t *)malloc(rank * sizeof(hsize_t));


        if (t >= 0)
        {
            /* dataspace parameters */
            (*field)->dims[t] = tdim;
            (*field)->maxdims[t] = H5S_UNLIMITED;
            (*field)->chunkdims[t] = 1;

            /* hyperslab selection parameters */
            (*field)->offset[t] = -1;   // increment before using!
            (*field)->stride[t] = 1;
            (*field)->count[t]  = 1;
            (*field)->block[t]  = 1;
        }

        if (x >= 0)
        {
            /* dataspace parameters */
            (*field)->dims[x] = nodex;
            (*field)->maxdims[x] = nodex;
            if (t >= 0)
                (*field)->chunkdims[x] = nodex;

            /* hyperslab selection parameters */
            (*field)->offset[x] = px*(nx-1);
            (*field)->stride[x] = 1;
            (*field)->count[x]  = 1;
            (*field)->block[x]  = (px == nprocx-1) ? nx : nx-1;
        }

        if (y >= 0)
        {
            /* dataspace parameters */
            (*field)->dims[y] = nodey;
            (*field)->maxdims[y] = nodey;
            if (t >= 0)
                (*field)->chunkdims[y] = nodey;

            /* hyperslab selection parameters */
            (*field)->offset[y] = py*(ny-1);
            (*field)->stride[y] = 1;
            (*field)->count[y]  = 1;
            (*field)->block[y]  = (py == nprocy-1) ? ny : ny-1;
        }

        if (z >= 0)
        {
            /* dataspace parameters */
            (*field)->dims[z] = nodez;
            (*field)->maxdims[z] = nodez;
            if (t >= 0)
                (*field)->chunkdims[z] = nodez;

            /* hyperslab selection parameters */
            (*field)->offset[z] = pz*(nz-1);
            (*field)->stride[z] = 1;
            (*field)->count[z]  = 1;
            (*field)->block[z]  = (pz == nprocz-1) ? nz : nz-1;
        }

        if (c >= 0)
        {
            /* dataspace parameters */
            (*field)->dims[c] = cdim;
            (*field)->maxdims[c] = cdim;
            if (t >= 0)
                (*field)->chunkdims[c] = cdim;

            /* hyperslab selection parameters */
            (*field)->offset[c] = 0;
            (*field)->stride[c] = 1;
            (*field)->count[c]  = 1;
            (*field)->block[c]  = cdim;
        }

        /* count number of data points (skip time dimension) */
        (*field)->n = 1;
        for(dim = 0; dim < rank; dim++)
            if(dim != t)
                (*field)->n *= (*field)->block[dim];

        /* finally, allocate buffer */
        (*field)->data = (double *)malloc((*field)->n * sizeof(double));

        return 0;
    }

    return -1;
}

static herr_t h5create_field(hid_t loc_id,
                             const char *name,
                             const char *title,
                             field_t *field)
{
    herr_t status;

    status = h5create_dataset(loc_id, name, title, field->dtype, field->rank,
                              field->dims, field->maxdims, field->chunkdims);

    return status;
}

static herr_t h5create_coord(hid_t loc_id, field_t *field)
{
    h5create_dataset(loc_id, "coord", "node coordinates of cap",
                     field->dtype, field->rank, field->dims, field->maxdims,
                     NULL);
    return 0;
}

static herr_t h5create_velocity(hid_t loc_id, field_t *field)
{
    h5create_dataset(loc_id, "velocity", "velocity values at cap nodes",
                     field->dtype, field->rank, field->dims, field->maxdims,
                     field->chunkdims);
    return 0;
}

static herr_t h5create_temperature(hid_t loc_id, field_t *field)
{
    h5create_dataset(loc_id, "temperature", "temperature values at cap nodes",
                     field->dtype, field->rank, field->dims, field->maxdims,
                     field->chunkdims);
    return 0;
}

static herr_t h5create_viscosity(hid_t loc_id, field_t *field)
{
    h5create_dataset(loc_id, "viscosity", "viscosity values at cap nodes",
                     field->dtype, field->rank, field->dims, field->maxdims,
                     field->chunkdims);
    return 0;
}

static herr_t h5create_pressure(hid_t loc_id, field_t *field)
{
    h5create_dataset(loc_id, "pressure", "pressure values at cap nodes",
                     field->dtype, field->rank, field->dims, field->maxdims,
                     field->chunkdims);
    return 0;
}

static herr_t h5create_stress(hid_t loc_id, field_t *field)
{
    h5create_dataset(loc_id, "stress", "stress values at cap nodes",
                     field->dtype, field->rank, field->dims, field->maxdims,
                     field->chunkdims);
    return 0;
}

static herr_t h5create_surf_coord(hid_t loc_id, field_t *field)
{
    h5create_dataset(loc_id, "coord", "surface coordinates",
                     field->dtype, field->rank, field->dims, field->maxdims,
                     NULL);
    return 0;
}

static herr_t h5create_surf_velocity(hid_t loc_id, field_t *field)
{
    h5create_dataset(loc_id, "velocity", "surface velocity",
                     field->dtype, field->rank, field->dims, field->maxdims,
                     field->chunkdims);
    return 0;
}

static herr_t h5create_surf_heatflux(hid_t loc_id, field_t *field)
{
    h5create_dataset(loc_id, "heatflux", "surface heatflux",
                     field->dtype, field->rank, field->dims, field->maxdims,
                     field->chunkdims);
    return 0;
}

static herr_t h5create_surf_topography(hid_t loc_id, field_t *field)
{
    h5create_dataset(loc_id, "topography", "surface topography",
                     field->dtype, field->rank, field->dims, field->maxdims,
                     field->chunkdims);
    return 0;
}

static herr_t h5create_time(hid_t loc_id)
{
    hid_t dataset;      /* dataset identifier */
    hid_t datatype;     /* row datatype identifier */
    hid_t dataspace;    /* memory dataspace */
    hid_t filespace;    /* file dataspace */
    hid_t dcpl_id;      /* dataset creation property list identifier */
    herr_t status;

    hsize_t dim = 0;
    hsize_t maxdim = H5S_UNLIMITED;
    hsize_t chunkdim = 1;

    /* Create the memory data type */
    datatype = H5Tcreate(H5T_COMPOUND, sizeof(struct HDF5_TIME));
    status = H5Tinsert(datatype, "step", HOFFSET(struct HDF5_TIME, step), H5T_NATIVE_INT);
    status = H5Tinsert(datatype, "time", HOFFSET(struct HDF5_TIME, time), H5T_NATIVE_FLOAT);
    status = H5Tinsert(datatype, "time_step", HOFFSET(struct HDF5_TIME, time_step), H5T_NATIVE_FLOAT);
    status = H5Tinsert(datatype, "cpu", HOFFSET(struct HDF5_TIME, cpu), H5T_NATIVE_FLOAT);
    status = H5Tinsert(datatype, "cpu_step", HOFFSET(struct HDF5_TIME, cpu_step), H5T_NATIVE_FLOAT);

    /* Create the dataspace */
    dataspace = H5Screate_simple(1, &dim, &maxdim);

    /* Modify dataset creation properties (enable chunking) */
    dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    status  = H5Pset_chunk(dcpl_id, 1, &chunkdim);

    /* Create the dataset */
    dataset = H5Dcreate(loc_id, "time", datatype, dataspace, dcpl_id);

    /*
     * Write necessary attributes for PyTables compatibility
     */

    set_attribute_string(dataset, "TITLE", "Timing table");
    set_attribute_string(dataset, "CLASS", "TABLE");
    set_attribute_string(dataset, "FLAVOR", "numpy");
    set_attribute_string(dataset, "VERSION", "2.6");

    set_attribute_llong(dataset, "NROWS", 0);

    set_attribute_string(dataset, "FIELD_0_NAME", "time");
    set_attribute_string(dataset, "FIELD_1_NAME", "time_step");
    set_attribute_string(dataset, "FIELD_2_NAME", "cpu");
    set_attribute_string(dataset, "FIELD_3_NAME", "cpu_step");

    set_attribute_double(dataset, "FIELD_0_FILL", 0);
    set_attribute_double(dataset, "FIELD_1_FILL", 0);
    set_attribute_double(dataset, "FIELD_2_FILL", 0);
    set_attribute_double(dataset, "FIELD_3_FILL", 0);

    set_attribute_int(dataset, "AUTOMATIC_INDEX", 1);
    set_attribute_int(dataset, "REINDEX", 1);
    set_attribute_string(dataset, "FILTERS_INDEX", FILTERS_P);

    /* Release resources */
    status = H5Pclose(dcpl_id);
    status = H5Sclose(dataspace);
    status = H5Tclose(datatype);
    status = H5Dclose(dataset);

    return 0;
}



static herr_t h5write_dataset(hid_t dset_id,
                              hid_t mem_type_id,
                              const void *data,
                              int rank,
                              hsize_t *memdims,
                              hsize_t *offset,
                              hsize_t *stride,
                              hsize_t *count,
                              hsize_t *block)
{
    hid_t memspace;     /* memory dataspace */
    hid_t filespace;    /* file dataspace */
    hid_t dxpl_id;      /* dataset transfer property list identifier */
    herr_t status;

    /* DEBUG
    if(size != NULL)
        printf("\th5write_dataset():\n"
               "\t\trank    = %d\n"
               "\t\tsize    = {%d,%d,%d,%d,%d}\n"
               "\t\tmemdims = {%d,%d,%d,%d,%d}\n"
               "\t\toffset  = {%d,%d,%d,%d,%d}\n"
               "\t\tstride  = {%d,%d,%d,%d,%d}\n"
               "\t\tblock   = {%d,%d,%d,%d,%d}\n",
               rank,
               (int)size[0], (int)size[1],
               (int)size[2], (int)size[3], (int)size[4],
               (int)memdims[0], (int)memdims[1],
               (int)memdims[2], (int)memdims[3], (int)memdims[4],
               (int)offset[0], (int)offset[1],
               (int)offset[2], (int)offset[3], (int)offset[4],
               (int)count[0], (int)count[1],
               (int)count[2], (int)count[3], (int)count[4],
               (int)block[0], (int)block[1],
               (int)block[2], (int)block[3], (int)block[4]);
    else
        printf("\th5write_dataset():\n"
               "\t\trank    = %d\n"
               "\t\tsize    = NULL\n"
               "\t\tmemdims = {%d,%d,%d,%d,%d}\n"
               "\t\toffset  = {%d,%d,%d,%d,%d}\n"
               "\t\tstride  = {%d,%d,%d,%d,%d}\n"
               "\t\tblock   = {%d,%d,%d,%d,%d}\n",
               rank,
               (int)memdims[0], (int)memdims[1],
               (int)memdims[2], (int)memdims[3], (int)memdims[4],
               (int)offset[0], (int)offset[1],
               (int)offset[2], (int)offset[3], (int)offset[4],
               (int)count[0], (int)count[1],
               (int)count[2], (int)count[3], (int)count[4],
               (int)block[0], (int)block[1],
               (int)block[2], (int)block[3], (int)block[4]);
    // */

    /* create memory dataspace */
    memspace = H5Screate_simple(rank, memdims, NULL);

    /* get file dataspace */
    filespace = H5Dget_space(dset_id);

    /* hyperslab selection */
    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                 offset, stride, count, block);

    /* dataset transfer property list */
    dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    status = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_INDEPENDENT);

    /* write the data to the hyperslab */
    status = H5Dwrite(dset_id, mem_type_id, memspace, filespace, dxpl_id, data);

    /* release resources */
    status = H5Pclose(dxpl_id);
    status = H5Sclose(filespace);
    status = H5Sclose(memspace);

    return 0;
}

static herr_t h5write_field(hid_t dset_id, field_t *field)
{
    herr_t status;

    status = h5write_dataset(dset_id, H5T_NATIVE_DOUBLE, field->data,
                             field->rank, field->block, field->offset,
                             field->stride, field->count, field->block);
    return status;
}


static herr_t h5close_field(field_t **field)
{
    if (field != NULL)
        if (*field != NULL)
        {
            free((*field)->dims);
            free((*field)->maxdims);
            if((*field)->chunkdims != NULL)
                free((*field)->chunkdims);
            free((*field)->offset);
            free((*field)->stride);
            free((*field)->count);
            free((*field)->block);
            free((*field)->data);
            free(*field);
        }
}

#endif


/****************************************************************************
 * Functions to save specific physical quantities as HDF5 arrays.           *
 ****************************************************************************/

#ifdef USE_HDF5

/* This function extends the time dimension of all time-varying field
 * objects. Before any data to a dataset, one should perform a collective
 * call to H5Dextend(dataset, field->dims).
 */
void h5extend_time_dimension(struct All_variables *E)
{
    int i;
    field_t *field[6];

    E->hdf5.count += 1;

    field[0] = E->hdf5.tensor3d;
    field[1] = E->hdf5.vector3d;
    field[2] = E->hdf5.vector2d;
    field[3] = E->hdf5.scalar3d;
    field[4] = E->hdf5.scalar2d;
    field[5] = E->hdf5.scalar1d;

    for(i = 0; i < 6; i++)
    {
        /* increase extent of time dimension in file dataspace */
        field[i]->dims[0] += 1;
        field[i]->maxdims[0] += 1;

        /* increment hyperslab offset */
        field[i]->offset[0] += 1;
    }
}

void h5output_coord(struct All_variables *E)
{
    hid_t cap_group;
    hid_t dataset;
    herr_t status;

    field_t *field;

    int i, j, k;
    int n, nx, ny, nz;
    int m, mx, my, mz;

    field = E->hdf5.const_vector3d;

    nx = E->lmesh.nox;
    ny = E->lmesh.noy;
    nz = E->lmesh.noz;

    mx = field->block[0];
    my = field->block[1];
    mz = field->block[2];

    /* prepare the data -- change citcom yxz order to xyz order */

    for(i = 0; i < mx; i++)
    {
        for(j = 0; j < my; j++)
        {
            for(k = 0; k < mz; k++)
            {
                n = k + i*nz + j*nz*nx;
                m = k + j*mz + i*mz*my;
                field->data[3*m+0] = E->sx[1][1][n+1];
                field->data[3*m+1] = E->sx[1][2][n+1];
                field->data[3*m+2] = E->sx[1][3][n+1];
            }
        }
    }

    /* write to dataset */
    cap_group = E->hdf5.cap_group;
    dataset = H5Dopen(cap_group, "coord");
    status = h5write_field(dataset, field);

    /* release resources */
    status = H5Dclose(dataset);
}

void h5output_velocity(struct All_variables *E, int cycles)
{
    int cap;
    hid_t cap_group;
    hid_t dataset;
    herr_t status;

    field_t *field;

    int i, j, k;
    int n, nx, ny, nz;
    int m, mx, my, mz;


    field = E->hdf5.vector3d;

    nx = E->lmesh.nox;
    ny = E->lmesh.noy;
    nz = E->lmesh.noz;

    mx = field->block[1];
    my = field->block[2];
    mz = field->block[3];

    /* extend all velocity fields -- collective I/O call */
    for(cap = 0; cap < E->sphere.caps; cap++)
    {
        cap_group = E->hdf5.cap_groups[cap];
        dataset   = H5Dopen(cap_group, "velocity");
        status    = H5Dextend(dataset, field->dims);
        status    = H5Dclose(dataset);
    }

    /* prepare the data -- change citcom yxz order to xyz order */
    for(i = 0; i < mx; i++)
    {
        for(j = 0; j < my; j++)
        {
            for(k = 0; k < mz; k++)
            {
                n = k + i*nz + j*nz*nx;
                m = k + j*mz + i*mz*my;
                field->data[3*m+0] = E->sphere.cap[1].V[1][n+1];
                field->data[3*m+1] = E->sphere.cap[1].V[2][n+1];
                field->data[3*m+2] = E->sphere.cap[1].V[3][n+1];
            }
        }
    }

    /* write to dataset */
    cap_group = E->hdf5.cap_group;
    dataset = H5Dopen(cap_group, "velocity");
    status = h5write_field(dataset, field);

    /* release resources */
    status = H5Dclose(dataset);
}

void h5output_temperature(struct All_variables *E, int cycles)
{
    int cap;
    hid_t cap_group;
    hid_t dataset;
    herr_t status;

    field_t *field;

    int i, j, k;
    int n, nx, ny, nz;
    int m, mx, my, mz;


    field = E->hdf5.scalar3d;

    nx = E->lmesh.nox;
    ny = E->lmesh.noy;
    nz = E->lmesh.noz;

    mx = field->block[1];
    my = field->block[2];
    mz = field->block[3];

    /* extend all temperature fields -- collective I/O call */
    for(cap = 0; cap < E->sphere.caps; cap++)
    {
        cap_group = E->hdf5.cap_groups[cap];
        dataset   = H5Dopen(cap_group, "temperature");
        status    = H5Dextend(dataset, field->dims);
        status    = H5Dclose(dataset);
    }

    /* prepare the data -- change citcom yxz order to xyz order */
    for(i = 0; i < mx; i++)
    {
        for(j = 0; j < my; j++)
        {
            for(k = 0; k < mz; k++)
            {
                n = k + i*nz + j*nz*nx;
                m = k + j*mz + i*mz*my;
                field->data[m] = E->T[1][n+1];
            }
        }
    }

    /* write to dataset */
    cap_group = E->hdf5.cap_group;
    dataset = H5Dopen(cap_group, "temperature");
    status = h5write_field(dataset, field);

    /* release resources */
    status = H5Dclose(dataset);
}

void h5output_viscosity(struct All_variables *E, int cycles)
{
    int cap;
    hid_t cap_group;
    hid_t dataset;
    herr_t status;

    field_t *field;

    int lev;
    int i, j, k;
    int n, nx, ny, nz;
    int m, mx, my, mz;


    field = E->hdf5.scalar3d;

    lev = E->mesh.levmax;

    nx = E->lmesh.nox;
    ny = E->lmesh.noy;
    nz = E->lmesh.noz;

    mx = field->block[1];
    my = field->block[2];
    mz = field->block[3];

    /* extend all viscosity fields -- collective I/O call */
    for(cap = 0; cap < E->sphere.caps; cap++)
    {
        cap_group = E->hdf5.cap_groups[cap];
        dataset   = H5Dopen(cap_group, "viscosity");
        status    = H5Dextend(dataset, field->dims);
        status    = H5Dclose(dataset);
    }

    /* prepare the data -- change citcom yxz order to xyz order */
    for(i = 0; i < mx; i++)
    {
        for(j = 0; j < my; j++)
        {
            for(k = 0; k < mz; k++)
            {
                n = k + i*nz + j*nz*nx;
                m = k + j*mz + i*mz*my;
                field->data[m] = E->VI[lev][1][n+1];
            }
        }
    }

    /* write to dataset */
    cap_group = E->hdf5.cap_group;
    dataset = H5Dopen(cap_group, "viscosity");
    status = h5write_field(dataset, field);

    /* release resources */
    status = H5Dclose(dataset);
}

void h5output_pressure(struct All_variables *E, int cycles)
{
}

void h5output_stress(struct All_variables *E, int cycles)
{
}

void h5output_material(struct All_variables *E)
{
}

void h5output_tracer(struct All_variables *E, int cycles)
{
}

void h5output_surf_botm(struct All_variables *E, int cycles)
{
    if(E->output.surf == 1) {
    }
    if(E->output.botm == 1) {
    }
}

void h5output_average(struct All_variables *E, int cycles)
{
}

void h5output_time(struct All_variables *E, int cycles)
{
    double CPU_time0();

    hid_t dataset;      /* dataset identifier */
    hid_t datatype;     /* row datatype identifier */
    hid_t dataspace;    /* memory dataspace */
    hid_t filespace;    /* file dataspace */
    hid_t dxpl_id;      /* data transfer property list identifier */

    herr_t status;

    hsize_t dim;
    hsize_t offset;
    hsize_t count;

    struct HDF5_TIME row;

    double current_time = CPU_time0();


    /* Prepare data */
    row.step = cycles;
    row.time = E->monitor.elapsed_time;
    row.time_step = E->advection.timestep;
    row.cpu = current_time - E->monitor.cpu_time_at_start;
    row.cpu_step = current_time - E->monitor.cpu_time_at_last_cycle;

    /* Get dataset */
    dataset = H5Dopen(E->hdf5.file_id, "time");

    /* Extend dataset -- note this is a collective call */
    dim = E->hdf5.count;
    status = H5Dextend(dataset, &dim);

    /* Get datatype */
    datatype = H5Dget_type(dataset);

    /* Define memory dataspace */
    dim = 1;
    dataspace = H5Screate_simple(1, &dim, NULL);

    /* Get file dataspace */
    filespace = H5Dget_space(dataset);

    /* Hyperslab selection parameters */
    offset = E->hdf5.count-1;
    count  = 1;

    /* Select hyperslab in file dataspace */
    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                 &offset, NULL, &count, NULL);

    /* Create property list for independent dataset write */
    dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    status = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_INDEPENDENT);

    /* Write to hyperslab selection */
    if (E->parallel.me == 0)
        status = H5Dwrite(dataset, datatype, dataspace, filespace,
                          dxpl_id, &row);

    /* Update NROWS attribute (for PyTables) */
    set_attribute_llong(dataset, "NROWS", E->hdf5.count);

    /* Release resources */
    status = H5Pclose(dxpl_id);
    status = H5Sclose(filespace);
    status = H5Sclose(dataspace);
    status = H5Tclose(datatype);
    status = H5Dclose(dataset);
}

void h5output_meta(struct All_variables *E)
{
    hid_t input;
    herr_t status;

    int n;
    int rank;
    hsize_t *dims;
    double *data;

    input = h5create_group(E->hdf5.file_id, "input", (size_t)0);

    /*
     * Advection_diffusion.inventory
     */

    status = set_attribute(input, "inputdiffusivity", H5T_NATIVE_FLOAT, &(E->control.inputdiff));

    status = set_attribute(input, "ADV", H5T_NATIVE_INT, &(E->advection.ADVECTION));
    status = set_attribute(input, "fixed_timestep", H5T_NATIVE_FLOAT, &(E->advection.fixed_timestep));
    status = set_attribute(input, "finetunedt", H5T_NATIVE_FLOAT, &(E->advection.fine_tune_dt));

    status = set_attribute(input, "adv_sub_iterations", H5T_NATIVE_INT, &(E->advection.temp_iterations));
    status = set_attribute(input, "maxadvtime", H5T_NATIVE_FLOAT, &(E->advection.max_dimensionless_time));

    status = set_attribute(input, "aug_lagr", H5T_NATIVE_INT, &(E->control.augmented_Lagr));
    status = set_attribute(input, "aug_number", H5T_NATIVE_DOUBLE, &(E->control.augmented));

    status = set_attribute(input, "filter_temp", H5T_NATIVE_INT, &(E->control.filter_temperature));

    /*
     * BC.inventory
     */

    status = set_attribute(input, "side_sbcs", H5T_NATIVE_INT, &(E->control.side_sbcs));

    status = set_attribute(input, "topvbc", H5T_NATIVE_INT, &(E->mesh.topvbc));
    status = set_attribute(input, "topvbxval", H5T_NATIVE_FLOAT, &(E->control.VBXtopval));
    status = set_attribute(input, "topvbyval", H5T_NATIVE_FLOAT, &(E->control.VBYtopval));

    status = set_attribute(input, "pseudo_free_surf", H5T_NATIVE_INT, &(E->control.pseudo_free_surf));

    status = set_attribute(input, "botvbc", H5T_NATIVE_INT, &(E->mesh.botvbc));
    status = set_attribute(input, "botvbxval", H5T_NATIVE_FLOAT, &(E->control.VBXbotval));
    status = set_attribute(input, "botvbyval", H5T_NATIVE_FLOAT, &(E->control.VBYbotval));

    status = set_attribute(input, "toptbc", H5T_NATIVE_INT, &(E->mesh.toptbc));
    status = set_attribute(input, "toptbcval", H5T_NATIVE_FLOAT, &(E->control.TBCtopval));

    status = set_attribute(input, "bottbc", H5T_NATIVE_INT, &(E->mesh.bottbc));
    status = set_attribute(input, "bottbcval", H5T_NATIVE_FLOAT, &(E->control.TBCbotval));

    status = set_attribute(input, "temperature_bound_adj", H5T_NATIVE_INT, &(E->control.temperature_bound_adj));
    status = set_attribute(input, "depth_bound_adj", H5T_NATIVE_FLOAT, &(E->control.depth_bound_adj));
    status = set_attribute(input, "width_bound_adj", H5T_NATIVE_FLOAT, &(E->control.width_bound_adj));

    /*
     * Const.inventory
     */

    status = set_attribute(input, "density", H5T_NATIVE_FLOAT, &(E->data.density));
    status = set_attribute(input, "thermdiff", H5T_NATIVE_FLOAT, &(E->data.therm_diff));
    status = set_attribute(input, "gravacc", H5T_NATIVE_FLOAT, &(E->data.grav_acc));
    status = set_attribute(input, "thermexp", H5T_NATIVE_FLOAT, &(E->data.therm_exp));
    status = set_attribute(input, "refvisc", H5T_NATIVE_FLOAT, &(E->data.ref_viscosity));
    status = set_attribute(input, "cp", H5T_NATIVE_FLOAT, &(E->data.Cp));
    status = set_attribute(input, "wdensity", H5T_NATIVE_FLOAT, &(E->data.density_above));
    status = set_attribute(input, "surftemp", H5T_NATIVE_FLOAT, &(E->data.surf_temp));
    status = set_attribute(input, "z_lith", H5T_NATIVE_FLOAT, &(E->viscosity.zlith));
    status = set_attribute(input, "z_410", H5T_NATIVE_FLOAT, &(E->viscosity.z410));
    status = set_attribute(input, "z_lmantle", H5T_NATIVE_FLOAT, &(E->viscosity.zlm));
    status = set_attribute(input, "z_cmb", H5T_NATIVE_FLOAT, &(E->viscosity.zcmb));
    status = set_attribute(input, "layer_km", H5T_NATIVE_FLOAT, &(E->data.layer_km));
    status = set_attribute(input, "radius_km", H5T_NATIVE_FLOAT, &(E->data.radius_km));
    status = set_attribute(input, "scalev", H5T_NATIVE_FLOAT, &(E->data.scalev));
    status = set_attribute(input, "scalet", H5T_NATIVE_FLOAT, &(E->data.scalet));

    /*
     * IC.inventory
     */

    status = set_attribute(input, "restart", H5T_NATIVE_INT, &(E->control.restart));
    status = set_attribute(input, "post_p", H5T_NATIVE_INT, &(E->control.post_p));
    status = set_attribute(input, "solution_cycles_init", H5T_NATIVE_INT, &(E->monitor.solution_cycles_init));
    status = set_attribute(input, "zero_elapsed_time", H5T_NATIVE_INT, &(E->control.zero_elapsed_time));

    status = set_attribute(input, "tic_method", H5T_NATIVE_INT, &(E->convection.tic_method));

    if (E->convection.tic_method == 0)
    {
        n = E->convection.number_of_perturbations;
        status = set_attribute(input, "num_perturbations", H5T_NATIVE_INT, &n);
        status = set_attribute_vector(input, "perturbl", n, H5T_NATIVE_INT, E->convection.perturb_ll);
        status = set_attribute_vector(input, "perturbm", n, H5T_NATIVE_INT, E->convection.perturb_mm);
        status = set_attribute_vector(input, "perturblayer", n, H5T_NATIVE_INT, E->convection.load_depth);
        status = set_attribute_vector(input, "perturbmag", n, H5T_NATIVE_FLOAT, E->convection.perturb_mag);
    }
    else if (E->convection.tic_method == 1)
    {
        status = set_attribute(input, "half_space_age", H5T_NATIVE_FLOAT, &(E->convection.half_space_age));
    }
    else if (E->convection.tic_method == 2)
    {
        status = set_attribute(input, "half_space_age", H5T_NATIVE_FLOAT, &(E->convection.half_space_age));
        status = set_attribute_vector(input, "blob_center", 3, H5T_NATIVE_FLOAT, E->convection.blob_center);
        status = set_attribute(input, "blob_radius", H5T_NATIVE_FLOAT, &(E->convection.blob_radius));
        status = set_attribute(input, "blob_dT", H5T_NATIVE_FLOAT, &(E->convection.blob_dT));
    }

    /*
     * Param.inventory
     */

    status = set_attribute(input, "file_vbcs", H5T_NATIVE_INT, &(E->control.vbcs_file));
    status = set_attribute_string(input, "vel_bound_file", E->control.velocity_boundary_file);

    status = set_attribute(input, "mat_control", H5T_NATIVE_INT, &(E->control.mat_control));
    status = set_attribute_string(input, "mat_file", E->control.mat_file);

    status = set_attribute(input, "lith_age", H5T_NATIVE_INT, &(E->control.lith_age));
    status = set_attribute_string(input, "lith_age_file", E->control.lith_age_file);
    status = set_attribute(input, "lith_age_time", H5T_NATIVE_INT, &(E->control.lith_age_time));
    status = set_attribute(input, "lith_age_depth", H5T_NATIVE_FLOAT, &(E->control.lith_age_depth));
    status = set_attribute(input, "mantle_temp", H5T_NATIVE_FLOAT, &(E->control.lith_age_mantle_temp));

    status = set_attribute(input, "start_age", H5T_NATIVE_FLOAT, &(E->control.start_age));
    status = set_attribute(input, "reset_startage", H5T_NATIVE_INT, &(E->control.reset_startage));

    /*
     * Phase.inventory
     */

    status = set_attribute(input, "Ra_410", H5T_NATIVE_FLOAT, &(E->control.Ra_410));
    status = set_attribute(input, "clapeyron410", H5T_NATIVE_FLOAT, &(E->control.clapeyron410));
    status = set_attribute(input, "transT410", H5T_NATIVE_FLOAT, &(E->control.transT410));
    status = set_attribute(input, "width410", H5T_NATIVE_FLOAT, &(E->control.width410));

    status = set_attribute(input, "Ra_670", H5T_NATIVE_FLOAT, &(E->control.Ra_670));
    status = set_attribute(input, "clapeyron670", H5T_NATIVE_FLOAT, &(E->control.clapeyron670));
    status = set_attribute(input, "transT670", H5T_NATIVE_FLOAT, &(E->control.transT670));
    status = set_attribute(input, "width670", H5T_NATIVE_FLOAT, &(E->control.width670));

    status = set_attribute(input, "Ra_cmb", H5T_NATIVE_FLOAT, &(E->control.Ra_cmb));
    status = set_attribute(input, "clapeyroncmb", H5T_NATIVE_FLOAT, &(E->control.clapeyroncmb));
    status = set_attribute(input, "transTcmb", H5T_NATIVE_FLOAT, &(E->control.transTcmb));
    status = set_attribute(input, "widthcmb", H5T_NATIVE_FLOAT, &(E->control.widthcmb));

    /*
     * Solver.inventory
     */

    status = set_attribute_string(input, "datafile", E->control.data_file);
    status = set_attribute_string(input, "datafile_old", E->control.old_P_file);

    status = set_attribute(input, "rayleigh", H5T_NATIVE_FLOAT, &(E->control.Atemp));
    status = set_attribute(input, "Q0", H5T_NATIVE_FLOAT, &(E->control.Q0));

    status = set_attribute(input, "stokes_flow_only", H5T_NATIVE_INT, &(E->control.stokes));

    status = set_attribute_string(input, "output_format", E->output.format);
    status = set_attribute_string(input, "output_optional", E->output.optional);
    status = set_attribute(input, "verbose", H5T_NATIVE_INT, &(E->control.verbose));
    status = set_attribute(input, "see_convergence", H5T_NATIVE_INT, &(E->control.print_convergence));

    /*
     * Sphere.inventory
     */

    status = set_attribute(input, "nproc_surf", H5T_NATIVE_INT, &(E->parallel.nprocxy));
    status = set_attribute(input, "nprocx", H5T_NATIVE_INT, &(E->parallel.nprocx));
    status = set_attribute(input, "nprocy", H5T_NATIVE_INT, &(E->parallel.nprocy));
    status = set_attribute(input, "nprocz", H5T_NATIVE_INT, &(E->parallel.nprocz));

    status = set_attribute(input, "coor", H5T_NATIVE_INT, &(E->control.coor));
    status = set_attribute_string(input, "coor_file", E->control.coor_file);

    status = set_attribute(input, "nodex", H5T_NATIVE_INT, &(E->mesh.nox));
    status = set_attribute(input, "nodey", H5T_NATIVE_INT, &(E->mesh.noy));
    status = set_attribute(input, "nodez", H5T_NATIVE_INT, &(E->mesh.noz));

    status = set_attribute(input, "levels", H5T_NATIVE_INT, &(E->mesh.levels));
    status = set_attribute(input, "mgunitx", H5T_NATIVE_INT, &(E->mesh.mgunitx));
    status = set_attribute(input, "mgunity", H5T_NATIVE_INT, &(E->mesh.mgunity));
    status = set_attribute(input, "mgunitz", H5T_NATIVE_INT, &(E->mesh.mgunitz));

    status = set_attribute(input, "radius_outer", H5T_NATIVE_DOUBLE, &(E->sphere.ro));
    status = set_attribute(input, "radius_inner", H5T_NATIVE_DOUBLE, &(E->sphere.ri));

    status = set_attribute(input, "caps", H5T_NATIVE_INT, &(E->sphere.caps));

    rank = 2;
    dims = (hsize_t *)malloc(rank * sizeof(hsize_t));
    dims[0] = E->sphere.caps;
    dims[1] = 4;
    data = (double *)malloc((dims[0]*dims[1]) * sizeof(double));

    for(n = 1; n <= E->sphere.caps; n++)
    {
        data[4*(n-1) + 0] = E->sphere.cap[n].theta[1];
        data[4*(n-1) + 1] = E->sphere.cap[n].theta[2];
        data[4*(n-1) + 2] = E->sphere.cap[n].theta[3];
        data[4*(n-1) + 3] = E->sphere.cap[n].theta[4];
    }
    status = set_attribute_array(input, "theta", rank, dims, H5T_NATIVE_DOUBLE, data);

    for(n = 1; n <= E->sphere.caps; n++)
    {
        data[4*(n-1) + 0] = E->sphere.cap[n].fi[1];
        data[4*(n-1) + 1] = E->sphere.cap[n].fi[2];
        data[4*(n-1) + 2] = E->sphere.cap[n].fi[3];
        data[4*(n-1) + 3] = E->sphere.cap[n].fi[4];
    }
    status = set_attribute_array(input, "fi", rank, dims, H5T_NATIVE_DOUBLE, data);

    free(data);
    free(dims);

    if (E->sphere.caps == 1)
    {
        status = set_attribute(input, "theta_min", H5T_NATIVE_DOUBLE, &(E->control.theta_min));
        status = set_attribute(input, "theta_max", H5T_NATIVE_DOUBLE, &(E->control.theta_max));
        status = set_attribute(input, "fi_min", H5T_NATIVE_DOUBLE, &(E->control.fi_min));
        status = set_attribute(input, "fi_max", H5T_NATIVE_DOUBLE, &(E->control.fi_max));
    }

    status = set_attribute(input, "ll_max", H5T_NATIVE_INT, &(E->sphere.llmax));
    status = set_attribute(input, "nlong", H5T_NATIVE_INT, &(E->sphere.noy));
    status = set_attribute(input, "nlati", H5T_NATIVE_INT, &(E->sphere.nox));
    status = set_attribute(input, "output_ll_max", H5T_NATIVE_INT, &(E->sphere.output_llmax));

    /*
     * Tracer.inventory
     */

    status = set_attribute(input, "tracer", H5T_NATIVE_INT, &(E->control.tracer));
    status = set_attribute_string(input, "tracer_file", E->control.tracer_file);

    /*
     * Visc.inventory
     */

    status = set_attribute_string(input, "Viscosity", E->viscosity.STRUCTURE);
    status = set_attribute(input, "visc_smooth_method", H5T_NATIVE_INT, &(E->viscosity.smooth_cycles));
    status = set_attribute(input, "VISC_UPDATE", H5T_NATIVE_INT, &(E->viscosity.update_allowed));

    n = E->viscosity.num_mat;
    status = set_attribute(input, "num_mat", H5T_NATIVE_INT, &n);
    status = set_attribute_vector(input, "visc0", n, H5T_NATIVE_FLOAT, E->viscosity.N0);
    status = set_attribute(input, "TDEPV", H5T_NATIVE_INT, &(E->viscosity.TDEPV));
    status = set_attribute(input, "rheol", H5T_NATIVE_INT, &(E->viscosity.RHEOL));
    status = set_attribute_vector(input, "viscE", n, H5T_NATIVE_FLOAT, E->viscosity.E);
    status = set_attribute_vector(input, "viscT", n, H5T_NATIVE_FLOAT, E->viscosity.T);
    status = set_attribute_vector(input, "viscZ", n, H5T_NATIVE_FLOAT, E->viscosity.Z);

    status = set_attribute(input, "SDEPV", H5T_NATIVE_INT, &(E->viscosity.SDEPV));
    status = set_attribute(input, "sdepv_misfit", H5T_NATIVE_FLOAT, &(E->viscosity.sdepv_misfit));
    status = set_attribute_vector(input, "sdepv_expt", n, H5T_NATIVE_FLOAT, E->viscosity.sdepv_expt);

    status = set_attribute(input, "VMIN", H5T_NATIVE_INT, &(E->viscosity.MIN));
    status = set_attribute(input, "visc_min", H5T_NATIVE_FLOAT, &(E->viscosity.min_value));

    status = set_attribute(input, "VMAX", H5T_NATIVE_INT, &(E->viscosity.MAX));
    status = set_attribute(input, "visc_max", H5T_NATIVE_FLOAT, &(E->viscosity.max_value));

    /*
     * Incompressible.inventory
     */

    status = set_attribute(input, "node_assemble", H5T_NATIVE_INT, &(E->control.NASSEMBLE));
    status = set_attribute(input, "precond", H5T_NATIVE_INT, &(E->control.precondition));

    status = set_attribute(input, "accuracy", H5T_NATIVE_DOUBLE, &(E->control.accuracy));
    status = set_attribute(input, "tole_compressibility", H5T_NATIVE_FLOAT, &(E->control.tole_comp));

    status = set_attribute(input, "mg_cycle", H5T_NATIVE_INT, &(E->control.mg_cycle));
    status = set_attribute(input, "down_heavy", H5T_NATIVE_INT, &(E->control.down_heavy));
    status = set_attribute(input, "up_heavy", H5T_NATIVE_INT, &(E->control.up_heavy));

    status = set_attribute(input, "vlowstep", H5T_NATIVE_INT, &(E->control.v_steps_low));
    status = set_attribute(input, "vhighstep", H5T_NATIVE_INT, &(E->control.v_steps_high));
    status = set_attribute(input, "piterations", H5T_NATIVE_INT, &(E->control.p_iterations));

    /* status = set_attribute(input, "", H5T_NATIVE_, &(E->)); */

    /*
     * Release resources
     */
    status = H5Gclose(input);
}

#endif
