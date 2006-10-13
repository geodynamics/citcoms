/*
 * Output_h5.c by Luis Armendariz and Eh Tan.
 * Copyright (C) 1994-2006, California Institute of Technology.
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
 */

/* Routines to write the output of the finite element cycles
 * into an HDF5 file, using parallel I/O.
 */


#include <stdlib.h>
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "parsing.h"
#include "output_h5.h"


#ifdef USE_HDF5

/****************************************************************************
 * Prototypes for functions local to this file. They are conditionally      *
 * included only when the HDF5 library is available.                        *
 ****************************************************************************/

/* for creation of HDF5 objects (wrapped for compatibility with PyTables) */
static hid_t h5create_file(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id);
static hid_t h5create_group(hid_t loc_id, const char *name, size_t size_hint);
static herr_t h5create_dataset(hid_t loc_id, const char *name, const char *title, hid_t type_id, int rank, hsize_t *dims, hsize_t *maxdims, hsize_t *chunkdims);

/* for creation of field and other dataset objects */
static herr_t h5allocate_field(struct All_variables *E, enum field_class_t field_class, int nsd, int time, hid_t dtype, field_t **field);
static herr_t h5create_field(hid_t loc_id, field_t *field, const char *name, const char *title);
static herr_t h5create_time(hid_t loc_id);
static herr_t h5create_connectivity(hid_t loc_id, int nel);

/* for writing to datasets */
static herr_t h5write_dataset(hid_t dset_id, hid_t mem_type_id, const void *data, int rank, hsize_t *memdims, hsize_t *offset, hsize_t *stride, hsize_t *count, hsize_t *block, int collective, int dowrite);
static herr_t h5write_field(hid_t dset_id, field_t *field, int collective, int dowrite);

/* for releasing resources from field object */
static herr_t h5close_field(field_t **field);

/* for writing to HDF5 attributes */
static herr_t find_attribute(hid_t loc_id, const char *attr_name);
herr_t set_attribute_string(hid_t obj_id, const char *attr_name, const char *attr_data);
herr_t set_attribute(hid_t obj_id, const char *attr_name, hid_t type_id, const void *data);
herr_t set_attribute_float(hid_t obj_id, const char *attr_name, float x);
herr_t set_attribute_double(hid_t obj_id, const char *attr_name, double x);
herr_t set_attribute_int(hid_t obj_id, const char *attr_name, int n);
herr_t set_attribute_long(hid_t obj_id, const char *attr_name, long n);
herr_t set_attribute_llong(hid_t obj_id, const char *attr_name, long long n);
herr_t set_attribute_array(hid_t obj_id, const char *attr_name, size_t rank, hsize_t *dims, hid_t type_id, const void *data);
herr_t set_attribute_vector(hid_t obj_id, const char *attr_name, hsize_t dim, hid_t type_id, const void *data);
herr_t set_attribute_int_vector(hid_t obj_id, const char *attr_name, hsize_t dim, const int *data);
herr_t set_attribute_float_vector(hid_t obj_id, const char *attr_name, hsize_t dim, const float *data);
herr_t set_attribute_double_vector(hid_t obj_id, const char *attr_name, hsize_t dim, const double *data);

/* constant data (only for first cycle) */
void h5output_meta(struct All_variables *);
void h5output_coord(struct All_variables *);
void h5output_surf_botm_coord(struct All_variables *);
void h5output_have_coord(struct All_variables *);
void h5output_material(struct All_variables *);
void h5output_connectivity(struct All_variables *);

/* time-varying data */
void h5extend_time_dimension(struct All_variables *);
void h5output_velocity(struct All_variables *, int);
void h5output_temperature(struct All_variables *, int);
void h5output_viscosity(struct All_variables *, int);
void h5output_pressure(struct All_variables *, int);
void h5output_stress(struct All_variables *, int);
void h5output_tracer(struct All_variables *, int);
void h5output_surf_botm(struct All_variables *, int);
void h5output_average(struct All_variables *, int);
void h5output_time(struct All_variables *, int);

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
        h5output_surf_botm_coord(E);
        h5output_have_coord(E);
        /*h5output_material(E);*/
        h5output_connectivity(E);
    }

    h5extend_time_dimension(E);
    h5output_velocity(E, cycles);
    h5output_temperature(E, cycles);
    h5output_viscosity(E, cycles);
    h5output_surf_botm(E, cycles);

    /* output tracer location if using tracer */
    if(E->control.tracer == 1)
        h5output_tracer(E, cycles);

    /* optional output below */
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
 * Function to read input parameters for legacy CitcomS                     *
 ****************************************************************************/

void h5input_params(struct All_variables *E)
{
#ifdef USE_HDF5
    int m = E->parallel.me;

    /* TODO: use non-optimized defaults to avoid unnecessary failures */

    input_int("cb_block_size", &(E->output.cb_block_size), "1048576", m);
    input_int("cb_buffer_size", &(E->output.cb_buffer_size), "4194304", m);

    input_int("sieve_buf_size", &(E->output.sieve_buf_size), "1048576", m);

    input_int("output_alignment", &(E->output.alignment), "262144", m);
    input_int("output_alignment_threshold", &(E->output.alignment_threshold), "524288", m);

    input_int("cache_mdc_nelmts", &(E->output.cache_mdc_nelmts), "10330", m);
    input_int("cache_rdcc_nelmts", &(E->output.cache_rdcc_nelmts), "521", m);
    input_int("cache_rdcc_nbytes", &(E->output.cache_rdcc_nbytes), "1048576", m);

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
    int nprocx = E->parallel.nprocx;
    int nprocy = E->parallel.nprocy;
    int nprocz = E->parallel.nprocz;
    int procs_per_cap;


    /*
     * Field variables
     */

    field_t *const_vector3d;
    field_t *const_vector2d;
    field_t *const_scalar1d;

    field_t *tensor3d;
    field_t *vector3d;
    field_t *vector2d;
    field_t *scalar3d;
    field_t *scalar2d;
    field_t *scalar1d;


    /*
     * MPI variables
     */
    
    MPI_Comm comm = E->parallel.world;
    MPI_Info info = MPI_INFO_NULL;
    int ierr;


    /*
     * HDF5 variables
     */

    hid_t file_id;      /* HDF5 file identifier */
    hid_t fcpl_id;      /* file creation property list identifier */
    hid_t fapl_id;      /* file access property list identifier */

    hid_t surf_group;   /* group identifier for top cap surface */
    hid_t botm_group;   /* group identifier for bottom cap surface */
    hid_t avg_group;    /* group identifier for horizontal averages */

    hid_t dtype;        /* datatype for dataset creation */

    herr_t status;


    /********************************************************************
     * Create HDF5 file using parallel I/O                              *
     ********************************************************************/

    /* TODO: figure out if it's possible give HDF5 a size hint when
     * creating the file
     */

    /* determine filename */
    snprintf(E->hdf5.filename, (size_t)100, "%s.h5", E->control.data_file);

    /* set up file creation property list with defaults */
    fcpl_id = H5P_DEFAULT;

    /* create an MPI_Info object to pass some tuning parameters
     * to the underlying MPI_File_open call
     */
    ierr = MPI_Info_create(&info);
    ierr = MPI_Info_set(info, "access_style", "write_once");
    ierr = MPI_Info_set(info, "collective_buffering", "true");
    ierr = MPI_Info_set(info, "cb_block_size", "1048576");
    ierr = MPI_Info_set(info, "cb_buffer_size", "4194304");

    /* set up file access property list with parallel I/O access */
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);

    status = H5Pset_sieve_buf_size(fapl_id, (size_t)(E->output.sieve_buf_size));
    status = H5Pset_alignment(fapl_id, (hsize_t)(E->output.alignment_threshold),
                                       (hsize_t)(E->output.alignment));
    status = H5Pset_cache(fapl_id, E->output.cache_mdc_nelmts,
                                   (size_t)(E->output.cache_rdcc_nelmts),
                                   (size_t)(E->output.cache_rdcc_nbytes),
                                   1.0);

    /* tell HDF5 to use MPI-IO */
    status  = H5Pset_fapl_mpio(fapl_id, comm, info);

    /* create a new file collectively and release property list identifier */
    file_id = h5create_file(E->hdf5.filename, H5F_ACC_TRUNC, fcpl_id, fapl_id);
    status  = H5Pclose(fapl_id);

    /* save the file identifier for later use */
    E->hdf5.file_id = file_id;
    E->hdf5.mpi_info = info;


    /********************************************************************
     * Allocate field objects for use in dataset writes...              *
     ********************************************************************/

    const_vector3d = NULL;
    const_vector2d = NULL;
    const_scalar1d = NULL;

    tensor3d = NULL;
    vector3d = NULL;
    vector2d = NULL;
    scalar3d = NULL;
    scalar2d = NULL;
    scalar1d = NULL;

    /* store solutions as floats in .h5 file */
    dtype = H5T_NATIVE_FLOAT;

    h5allocate_field(E, VECTOR_FIELD, 3, 0, dtype, &const_vector3d);
    h5allocate_field(E, VECTOR_FIELD, 2, 0, dtype, &const_vector2d);
    h5allocate_field(E, SCALAR_FIELD, 1, 0, dtype, &const_scalar1d);
    h5allocate_field(E, TENSOR_FIELD, 3, 1, dtype, &tensor3d);
    h5allocate_field(E, VECTOR_FIELD, 3, 1, dtype, &vector3d);
    h5allocate_field(E, VECTOR_FIELD, 2, 1, dtype, &vector2d);
    h5allocate_field(E, SCALAR_FIELD, 3, 1, dtype, &scalar3d);
    h5allocate_field(E, SCALAR_FIELD, 2, 1, dtype, &scalar2d);
    h5allocate_field(E, SCALAR_FIELD, 1, 1, dtype, &scalar1d);

    /* reuse largest buffer */
    if (E->output.stress == 1)
    {
        tensor3d->data = (float *)malloc((tensor3d->n) * sizeof(float));
        vector3d->data = tensor3d->data;
        vector2d->data = tensor3d->data;
        scalar3d->data = tensor3d->data;
        scalar2d->data = tensor3d->data;
        scalar1d->data = tensor3d->data;
        const_vector3d->data = tensor3d->data;
        const_vector2d->data = tensor3d->data;
        const_scalar1d->data = tensor3d->data;
    }
    else
    {
        tensor3d->data = NULL;
        vector3d->data = (float *)malloc((vector3d->n) * sizeof(float));
        vector2d->data = vector3d->data;
        scalar3d->data = vector3d->data;
        scalar2d->data = vector3d->data;
        scalar1d->data = vector3d->data;
        const_vector3d->data = vector3d->data;
        const_vector2d->data = vector3d->data;
        const_scalar1d->data = vector3d->data;

    }

    E->hdf5.const_vector3d = const_vector3d;
    E->hdf5.const_vector2d = const_vector2d;
    E->hdf5.const_scalar1d = const_scalar1d;

    E->hdf5.tensor3d = tensor3d;
    E->hdf5.vector3d = vector3d;
    E->hdf5.vector2d = vector2d;
    E->hdf5.scalar3d = scalar3d;
    E->hdf5.scalar2d = scalar2d;
    E->hdf5.scalar1d = scalar1d;

    E->hdf5.field[0] = tensor3d;
    E->hdf5.field[1] = vector3d;
    E->hdf5.field[2] = vector2d;
    E->hdf5.field[3] = scalar3d;
    E->hdf5.field[4] = scalar2d;
    E->hdf5.field[5] = scalar1d;


    /********************************************************************
     * Create time table (to store nondimensional and cpu times)        *
     ********************************************************************/
    h5create_time(file_id);

    /********************************************************************
     * Create necessary groups and arrays                               *
     ********************************************************************/

    h5create_field(file_id, const_vector3d, "coord", "coordinates of nodes");
    h5create_field(file_id, vector3d, "velocity", "velocity values on nodes");
    h5create_field(file_id, scalar3d, "temperature", "temperature values on nodes");
    h5create_field(file_id, scalar3d, "viscosity", "viscosity values on nodes");

    /* Create /pressure dataset */
    if (E->output.pressure == 1)
        h5create_field(file_id, scalar3d, "pressure", "pressure values on nodes");

    /* Create /stress dataset */
    if (E->output.stress == 1)
        h5create_field(file_id, tensor3d, "stress", "stress values on nodes");

    /* Create /connectivity dataset */
    procs_per_cap = nprocx * nprocy * nprocz;
    if (E->output.connectivity == 1)
        h5create_connectivity(file_id, E->lmesh.nel * procs_per_cap);

    /* Create /surf/ group*/
    if (E->output.surf == 1)
    {
        surf_group = h5create_group(file_id, "surf", (size_t)0);
        h5create_field(surf_group, const_vector2d, "coord", "top surface coordinates");
        h5create_field(surf_group, vector2d, "velocity", "top surface velocity");
        h5create_field(surf_group, scalar2d, "heatflux", "top surface heatflux");
        h5create_field(surf_group, scalar2d, "topography", "top surface topography");
        status = H5Gclose(surf_group);
    }

    /* Create /botm/ group */
    if (E->output.botm == 1)
    {
        botm_group = h5create_group(file_id, "botm", (size_t)0);
        h5create_field(botm_group, const_vector2d, "coord", "bottom surface coordinates");
        h5create_field(botm_group, vector2d, "velocity", "bottom surface velocity");
        h5create_field(botm_group, scalar2d, "heatflux", "bottom surface heatflux");
        h5create_field(botm_group, scalar2d, "topography", "bottom surface topography");
        status = H5Gclose(botm_group);
    }

    /* Create /horiz_avg/ group */
    if(E->output.average == 1)
    {
        avg_group = h5create_group(file_id, "horiz_avg", (size_t)0);
        h5create_field(avg_group, const_scalar1d, "coord", "radial coordinates of horizontal planes");
        h5create_field(avg_group, scalar1d, "temperature", "horizontal temperature average");
        h5create_field(avg_group, scalar1d, "velocity_xy", "horizontal Vxy average (rms)");
        h5create_field(avg_group, scalar1d, "velocity_z", "horizontal Vz average (rms)");
        status = H5Gclose(avg_group);
    }


    /* Remember number of times we have called h5output()
     * TODO: for restart, initialize to last known value */
    E->hdf5.count = 0;

    /* Determine current cap and remember it */
    cap = (E->parallel.me) / (nprocx * nprocy * nprocz);
    E->hdf5.cap = cap;


#endif
}

/* Finalizing access to HDF5 objects. Note that this function
 * needs to be visible to external files (even when HDF5 isn't
 * configured).
 */
void h5output_close(struct All_variables *E)
{
#ifdef USE_HDF5
    int ierr;
    herr_t status;

    /* close file */
    status = H5Fclose(E->hdf5.file_id);

    /* close mpi info object */
    ierr = MPI_Info_free(&(E->hdf5.mpi_info));

    /* free buffer attached to vector3d */
    free((E->hdf5.vector3d)->data);

    /* close fields (deallocate buffers) */
    h5close_field(&(E->hdf5.const_vector3d));
    h5close_field(&(E->hdf5.const_vector2d));
    h5close_field(&(E->hdf5.const_scalar1d));
    h5close_field(&(E->hdf5.tensor3d));
    h5close_field(&(E->hdf5.vector3d));
    h5close_field(&(E->hdf5.vector2d));
    h5close_field(&(E->hdf5.scalar3d));
    h5close_field(&(E->hdf5.scalar2d));
    h5close_field(&(E->hdf5.scalar1d));
#endif
}



/*****************************************************************************
 * Private functions to simplify certain tasks in the h5output_*() functions *
 * The rest of the file can now be hidden from the compiler, when HDF5       *
 * is not enabled.                                                           *
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

    /* create the dataspace for the dataset */
    dataspace = H5Screate_simple(rank, dims, maxdims);
    if (dataspace < 0)
    {
        /*TODO: print error*/
        return -1;
    }

    dcpl_id = H5P_DEFAULT;
    if (chunkdims != NULL)
    {
        /* modify dataset creation properties to enable chunking */
        dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
        status = H5Pset_chunk(dcpl_id, rank, chunkdims);
        /*status = H5Pset_fill_value(dcpl_id, H5T_NATIVE_FLOAT, &fillvalue);*/
    }

    /* create the dataset */
    dataset = H5Dcreate(loc_id, name, type_id, dataspace, dcpl_id);
    if (dataset < 0)
    {
        /*TODO: print error*/
        return -1;
    }

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
    int t = -100;   /* time dimension */
    int s = -100;   /* caps dimension */
    int x = -100;   /* first spatial dimension */
    int y = -100;   /* second spatial dimension */
    int z = -100;   /* third spatial dimension */
    int c = -100;   /* dimension for components */

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

    /* clear struct pointer */
    *field = NULL;

    /* start with caps as the first dimension */
    rank = 1;
    s = 0;

    /* add the spatial dimensions */
    switch (nsd)
    {
        case 3:
            rank += 3;
            x = 1;
            y = 2;
            z = 3;
            break;
        case 2:
            rank += 2;
            x = 1;
            y = 2;
            break;
        case 1:
            rank += 1;
            z = 1;
            break;
        default:
            return -1;
    }

    /* make time the first dimension (if necessary) */
    if (time > 0)
    {
        rank += 1;
        t  = 0;
        s += 1;
        x += 1;
        y += 1;
        z += 1;
    }

    /* add components dimension at end */
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

    if (rank > 1)
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
            (*field)->offset[t] = -1;   /* increment before using! */
            (*field)->stride[t] = 1;
            (*field)->count[t]  = 1;
            (*field)->block[t]  = 1;
        }

        if (s >= 0)
        {
            /* dataspace parameters */
            (*field)->dims[s] = E->sphere.caps;
            (*field)->maxdims[s] = E->sphere.caps;
            if (t >= 0)
                (*field)->chunkdims[s] = E->sphere.caps;
            
            /* hyperslab selection parameters */
            (*field)->offset[s] = E->parallel.me / (nprocx*nprocy*nprocz);
            (*field)->stride[s] = 1;
            (*field)->count[s]  = 1;
            (*field)->block[s]  = 1;
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
        /*(*field)->data = (float *)malloc((*field)->n * sizeof(float));*/

        return 0;
    }

    return -1;
}

static herr_t h5create_field(hid_t loc_id,
                             field_t *field,
                             const char *name,
                             const char *title)
{
    herr_t status;

    status = h5create_dataset(loc_id, name, title, field->dtype, field->rank,
                              field->dims, field->maxdims, field->chunkdims);

    return status;
}


static herr_t h5write_dataset(hid_t dset_id,
                              hid_t mem_type_id,
                              const void *data,
                              int rank,
                              hsize_t *memdims,
                              hsize_t *offset,
                              hsize_t *stride,
                              hsize_t *count,
                              hsize_t *block,
                              int collective,
                              int dowrite)
{
    hid_t memspace;     /* memory dataspace */
    hid_t filespace;    /* file dataspace */
    hid_t dxpl_id;      /* dataset transfer property list identifier */
    herr_t status;

    /* create memory dataspace */
    memspace = H5Screate_simple(rank, memdims, NULL);
    if (memspace < 0)
    {
        /*TODO: print error*/
        return -1;
    }

    /* get file dataspace */
    filespace = H5Dget_space(dset_id);
    if (filespace < 0)
    {
        /*TODO: print error*/
        H5Sclose(memspace);
        return -1;
    }

    /* hyperslab selection */
    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                 offset, stride, count, block);
    if (status < 0)
    {
        /*TODO: print error*/
        status = H5Sclose(filespace);
        status = H5Sclose(memspace);
        return -1;
    }

    /* dataset transfer property list */
    dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    if (dxpl_id < 0)
    {
        /*TODO: print error*/
        status = H5Sclose(filespace);
        status = H5Sclose(memspace);
        return -1;
    }

    if (collective)
        status = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);
    else
        status = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_INDEPENDENT);

    if (status < 0)
    {
        /*TODO: print error*/
        status = H5Pclose(dxpl_id);
        status = H5Sclose(filespace);
        status = H5Sclose(memspace);
        return -1;
    }

    /* write the data to the hyperslab */
    if (dowrite || collective)
    {
        status = H5Dwrite(dset_id, mem_type_id, memspace, filespace, dxpl_id, data);
        if (status < 0)
        {
            /*TODO: print error*/
            H5Pclose(dxpl_id);
            H5Sclose(filespace);
            H5Sclose(memspace);
            return -1;
        }
    }

    /* release resources */
    status = H5Pclose(dxpl_id);
    status = H5Sclose(filespace);
    status = H5Sclose(memspace);

    return 0;
}

static herr_t h5write_field(hid_t dset_id, field_t *field, int collective, int dowrite)
{
    herr_t status;

    status = h5write_dataset(dset_id, H5T_NATIVE_FLOAT, field->data,
                             field->rank, field->block, field->offset,
                             field->stride, field->count, field->block,
                             collective, dowrite);
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
            /*free((*field)->data);*/
            free(*field);
        }
}



/****************************************************************************
 * The following functions are used to save specific physical quantities    *
 * from CitcomS into HDF5 arrays.                                           *
 ****************************************************************************/

/* This function extends the time dimension of all time-varying field
 * objects. Before any data gets written to a dataset, one should perform
 * a collective call to H5Dextend(dataset, field->dims).
 */
void h5extend_time_dimension(struct All_variables *E)
{
    int i;
    E->hdf5.count += 1;
    for(i = 0; i < 6; i++)
    {
        /* check for optional fields */
        if (E->hdf5.field[i] == NULL)
            continue;

        /* increase extent of time dimension in file dataspace */
        (E->hdf5.field[i])->dims[0] += 1;

        /* increment hyperslab offset */
        (E->hdf5.field[i])->offset[0] += 1;
    }
}


/****************************************************************************
 * 3D Fields                                                                *
 ****************************************************************************/

void h5output_coord(struct All_variables *E)
{
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

    mx = field->block[1];
    my = field->block[2];
    mz = field->block[3];

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
    dataset = H5Dopen(E->hdf5.file_id, "/coord");
    status  = h5write_field(dataset, field, 1, 1);

    /* release resources */
    status = H5Dclose(dataset);
}

void h5output_velocity(struct All_variables *E, int cycles)
{
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

    mx = field->block[2];
    my = field->block[3];
    mz = field->block[4];

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
    dataset = H5Dopen(E->hdf5.file_id, "/velocity");
    status  = H5Dextend(dataset, field->dims);
    status  = h5write_field(dataset, field, 1, 1);

    /* release resources */
    status = H5Dclose(dataset);
}

void h5output_temperature(struct All_variables *E, int cycles)
{
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

    mx = field->block[2];
    my = field->block[3];
    mz = field->block[4];

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
    dataset = H5Dopen(E->hdf5.file_id, "/temperature");
    status  = H5Dextend(dataset, field->dims);
    status  = h5write_field(dataset, field, 1, 1);

    /* release resources */
    status = H5Dclose(dataset);
}

void h5output_viscosity(struct All_variables *E, int cycles)
{
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

    mx = field->block[2];
    my = field->block[3];
    mz = field->block[4];

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
    dataset = H5Dopen(E->hdf5.file_id, "/viscosity");
    status  = H5Dextend(dataset, field->dims);
    status  = h5write_field(dataset, field, 1, 1);

    /* release resources */
    status = H5Dclose(dataset);
}

void h5output_pressure(struct All_variables *E, int cycles)
{
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

    /* prepare the data -- change citcom yxz order to xyz order */
    for(i = 0; i < mx; i++)
    {
        for(j = 0; j < my; j++)
        {
            for(k = 0; k < mz; k++)
            {
                n = k + i*nz + j*nz*nx;
                m = k + j*mz + i*mz*my;
                field->data[m] = E->NP[1][n+1];
            }
        }
    }

    /* write to dataset */
    dataset = H5Dopen(E->hdf5.file_id, "/pressure");
    status  = H5Dextend(dataset, field->dims);
    status  = h5write_field(dataset, field, 1, 1);

    /* release resources */
    status = H5Dclose(dataset);
}

void h5output_stress(struct All_variables *E, int cycles)
{
    hid_t dataset;
    herr_t status;
    field_t *field;

    int i, j, k;
    int n, nx, ny, nz;
    int m, mx, my, mz;

    field = E->hdf5.tensor3d;

    nx = E->lmesh.nox;
    ny = E->lmesh.noy;
    nz = E->lmesh.noz;

    mx = field->block[1];
    my = field->block[2];
    mz = field->block[3];

    /* prepare the data -- change citcom yxz order to xyz order */
    for(i = 0; i < mx; i++)
    {
        for(j = 0; j < my; j++)
        {
            for(k = 0; k < mz; k++)
            {
                n = k + i*nz + j*nz*nx;
                m = k + j*mz + i*mz*my;
                field->data[6*m+0] = E->gstress[1][6*n+1];
                field->data[6*m+1] = E->gstress[1][6*n+2];
                field->data[6*m+2] = E->gstress[1][6*n+3];
                field->data[6*m+3] = E->gstress[1][6*n+4];
                field->data[6*m+4] = E->gstress[1][6*n+5];
                field->data[6*m+5] = E->gstress[1][6*n+6];
            }
        }
    }

    /* write to dataset */
    dataset = H5Dopen(E->hdf5.file_id, "/stress");
    status  = H5Dextend(dataset, field->dims);
    status  = h5write_field(dataset, field, 1, 1);

    /* release resources */
    status = H5Dclose(dataset);
}

void h5output_material(struct All_variables *E)
{
}

void h5output_tracer(struct All_variables *E, int cycles)
{
}

/****************************************************************************
 * 2D Fields                                                                *
 ****************************************************************************/

void h5output_surf_botm_coord(struct All_variables *E)
{
    hid_t dataset;
    herr_t status;
    field_t *field;

    int i, j, k;
    int n, nx, ny, nz;
    int m, mx, my;

    int pz = E->parallel.me_loc[3];
    int nprocz = E->parallel.nprocz;

    field = E->hdf5.const_vector2d;

    nx = E->lmesh.nox;
    ny = E->lmesh.noy;
    nz = E->lmesh.noz;

    mx = field->block[1];
    my = field->block[2];

    if (E->output.surf == 1)
    {
        k = nz-1;
        for(i = 0; i < mx; i++)
        {
            for(j = 0; j < my; j++)
            {
                n = k + i*nz + j*nz*nx;
                m = j + i*my;
                field->data[2*m+0] = E->sx[1][1][n+1];
                field->data[2*m+1] = E->sx[1][2][n+1];
            }
        }
        dataset = H5Dopen(E->hdf5.file_id, "/surf/coord");
        status = h5write_field(dataset, field, 0, (pz == nprocz-1));
        status = H5Dclose(dataset);
    }

    if (E->output.botm == 1)
    {
        k = 0;
        for(i = 0; i < mx; i++)
        {
            for(j = 0; j < my; j++)
            {
                n = k + i*nz + j*nz*nx;
                m = j + i*my;
                field->data[2*m+0] = E->sx[1][1][n+1];
                field->data[2*m+1] = E->sx[1][2][n+1];
            }
        }
        dataset = H5Dopen(E->hdf5.file_id, "/botm/coord");
        status = h5write_field(dataset, field, 0, (pz == 0));
        status = H5Dclose(dataset);
    }
}

void h5output_surf_botm(struct All_variables *E, int cycles)
{
    hid_t file_id;
    hid_t dataset;
    herr_t status;
    field_t *scalar;
    field_t *vector;

    float *topo;

    int i, j, k;
    int n, nx, ny, nz;
    int m, mx, my;

    int pz = E->parallel.me_loc[3];
    int nprocz = E->parallel.nprocz;

    file_id = E->hdf5.file_id;

    scalar = E->hdf5.scalar2d;
    vector = E->hdf5.vector2d;

    nx = E->lmesh.nox;
    ny = E->lmesh.noy;
    nz = E->lmesh.noz;

    mx = scalar->block[2];
    my = scalar->block[3];


    heat_flux(E);
    get_STD_topo(E, E->slice.tpg, E->slice.tpgb, E->slice.divg, E->slice.vort, cycles);


    /********************************************************************
     * Top surface                                                      *
     ********************************************************************/
    if (E->output.surf == 1)
    {
        /* radial index */
        k = nz-1;

        /* velocity data */
        for(i = 0; i < mx; i++)
        {
            for(j = 0; j < my; j++)
            {
                n = k + i*nz + j*nz*nx;
                m = j + i*my;
                vector->data[2*m+0] = E->sphere.cap[1].V[1][n+1];
                vector->data[2*m+1] = E->sphere.cap[1].V[2][n+1];
            }
        }
        dataset = H5Dopen(file_id, "/surf/velocity");
        status = H5Dextend(dataset, vector->dims);
        status = h5write_field(dataset, vector, 0, (pz == nprocz-1));
        status = H5Dclose(dataset);

        /* heatflux data */
        for(i = 0; i < mx; i++)
        {
            for(j = 0; j < my; j++)
            {
                n = k + i*nz + j*nz*nx;
                m = j + i*my;
                scalar->data[m] = E->slice.shflux[1][n+1];
            }
        }
        dataset = H5Dopen(file_id, "/surf/heatflux");
        status = H5Dextend(dataset, scalar->dims);
        status = h5write_field(dataset, scalar, 0, (pz == nprocz-1));
        status = H5Dclose(dataset);

        /* choose either STD topo or pseudo-free-surf topo */
        if (E->control.pseudo_free_surf)
            topo = E->slice.freesurf[1];
        else
            topo = E->slice.tpg[1];

        /* topography data */
        for(i = 0; i < mx; i++)
        {
            for(j = 0; j < my; j++)
            {
                n = k + i*nz + j*nz*nx;
                m = j + i*my;
                scalar->data[m] = topo[i];
            }
        }
        dataset = H5Dopen(file_id, "/surf/topography");
        status = H5Dextend(dataset, scalar->dims);
        status = h5write_field(dataset, scalar, 0, (pz == nprocz-1));
        status = H5Dclose(dataset);
    }


    /********************************************************************
     * Bottom surface                                                   *
     ********************************************************************/
    if (E->output.botm == 1)
    {
        /* radial index */
        k = 0;

        /* velocity data */
        for(i = 0; i < mx; i++)
        {
            for(j = 0; j < my; j++)
            {
                n = k + i*nz + j*nz*nx;
                m = j + i*my;
                vector->data[2*m+0] = E->sphere.cap[1].V[1][n+1];
                vector->data[2*m+1] = E->sphere.cap[1].V[2][n+1];
            }
        }
        dataset = H5Dopen(file_id, "/botm/velocity");
        status = H5Dextend(dataset, vector->dims);
        status = h5write_field(dataset, vector, 0, (pz == 0));
        status = H5Dclose(dataset);

        /* heatflux data */
        for(i = 0; i < mx; i++)
        {
            for(j = 0; j < my; j++)
            {
                n = k + i*nz + j*nz*nx;
                m = j + i*my;
                scalar->data[m] = E->slice.bhflux[1][n+1];
            }
        }
        dataset = H5Dopen(file_id, "/botm/heatflux");
        status = H5Dextend(dataset, scalar->dims);
        status = h5write_field(dataset, scalar, 0, (pz == 0));
        status = H5Dclose(dataset);

        /* topography data */
        topo = E->slice.tpg[1];
        for(i = 0; i < mx; i++)
        {
            for(j = 0; j < my; j++)
            {
                n = k + i*nz + j*nz*nx;
                m = j + i*my;
                scalar->data[m] = topo[i];
            }
        }
        dataset = H5Dopen(file_id, "/botm/topography");
        status = H5Dextend(dataset, scalar->dims);
        status = h5write_field(dataset, scalar, 0, (pz == 0));
        status = H5Dclose(dataset);
    }
}


/****************************************************************************
 * 1D Fields                                                                *
 ****************************************************************************/

void h5output_have_coord(struct All_variables *E)
{
    hid_t file_id;
    hid_t dataset;
    herr_t status;

    field_t *field;

    int k;
    int mz;

    int px = E->parallel.me_loc[1];
    int py = E->parallel.me_loc[2];

    field = E->hdf5.const_scalar1d;

    mz = field->block[1];

    if (E->output.average == 1)
    {
        for(k = 0; k < mz; k++)
            field->data[k] = E->sx[1][3][k+1];
        dataset = H5Dopen(E->hdf5.file_id, "/horiz_avg/coord");
        status = h5write_field(dataset, field, 0, (px == 0 && py == 0));
        status = H5Dclose(dataset);
    }

}

void h5output_average(struct All_variables *E, int cycles)
{
    /* horizontal average output of temperature and rms velocity */
    void compute_horiz_avg();

    hid_t file_id;
    hid_t dataset;
    herr_t status;

    field_t *field;

    int k;
    int mz;

    int px = E->parallel.me_loc[1];
    int py = E->parallel.me_loc[2];


    file_id = E->hdf5.file_id;

    field = E->hdf5.scalar1d;

    mz = field->block[2];

    /* calculate horizontal averages */
    compute_horiz_avg(E);

    /*
     * note that only the first nprocz processes need to output
     */

    /* temperature horizontal average */
    for(k = 0; k < mz; k++)
        field->data[k] = E->Have.T[k+1];
    dataset = H5Dopen(file_id, "/horiz_avg/temperature");
    status = H5Dextend(dataset, field->dims);
    status = h5write_field(dataset, field, 0, (px == 0 && py == 0));
    status = H5Dclose(dataset);

    /* Vxy horizontal average (rms) */
    for(k = 0; k < mz; k++)
        field->data[k] = E->Have.V[1][k+1];
    dataset = H5Dopen(file_id, "/horiz_avg/velocity_xy");
    status = H5Dextend(dataset, field->dims);
    status = h5write_field(dataset, field, 0, (px == 0 && py == 0));
    status = H5Dclose(dataset);

    /* Vz horizontal average (rms) */
    for(k = 0; k < mz; k++)
        field->data[k] = E->Have.V[2][k+1];
    dataset = H5Dopen(file_id, "/horiz_avg/velocity_z");
    status = H5Dextend(dataset, field->dims);
    status = h5write_field(dataset, field, 0, (px == 0 && py == 0));
    status = H5Dclose(dataset);
}


/****************************************************************************
 * Create and output /connectivity dataset                                  *
 ****************************************************************************/

static herr_t h5create_connectivity(hid_t loc_id, int nel)
{
    hid_t dataset;
    hid_t dataspace;
    herr_t status;

    hsize_t dims[2];

    dims[0] = nel;
    dims[1] = 8;

    /* Create the dataspace */
    dataspace = H5Screate_simple(2, dims, NULL);

    /* Create the dataset */
    dataset = H5Dcreate(loc_id, "connectivity", H5T_NATIVE_INT, dataspace, H5P_DEFAULT);

    /* Write necessary attributes for PyTables compatibility */
    set_attribute_string(dataset, "TITLE", "Node connectivity");
    set_attribute_string(dataset, "CLASS", "ARRAY");
    set_attribute_string(dataset, "FLAVOR", "numpy");
    set_attribute_string(dataset, "VERSION", "2.3");

    status = H5Sclose(dataspace);
    status = H5Dclose(dataset);
    return 0;
}

void h5output_connectivity(struct All_variables *E)
{
    hid_t dataset;
    herr_t status;

    int rank = 2;
    hsize_t memdims[2];
    hsize_t offset[2];
    hsize_t stride[2];
    hsize_t count[2];
    hsize_t block[2];

    int p;
    int px = E->parallel.me_loc[1];
    int py = E->parallel.me_loc[2];
    int pz = E->parallel.me_loc[3];
    int nprocx = E->parallel.nprocx;
    int nprocy = E->parallel.nprocy;
    int nprocz = E->parallel.nprocz;


    int e;
    int nel = E->lmesh.nel;
    int *ien;

    int *data;

    if (E->output.connectivity == 1)
    {
        /* process id (local to cap) */
        p = pz + px*nprocz + py*nprocz*nprocx;

        rank = 2;

        memdims[0] = nel;
        memdims[1] = 8;

        offset[0] = nel * p;
        offset[1] = 0;

        stride[0] = 1;
        stride[1] = 1;

        count[0] = 1;
        count[1] = 1;

        block[0] = nel;
        block[1] = 8;

        data = (int *)malloc((nel*8) * sizeof(int));

        for(e = 0; e < nel; e++)
        {
            ien = E->ien[1][e+1].node;
            data[8*e+0] = ien[1]-1; /* TODO: subtract one? */
            data[8*e+1] = ien[2]-1;
            data[8*e+2] = ien[3]-1;
            data[8*e+3] = ien[4]-1;
            data[8*e+4] = ien[5]-1;
            data[8*e+5] = ien[6]-1;
            data[8*e+6] = ien[7]-1;
            data[8*e+7] = ien[8]-1;
        }

        dataset = H5Dopen(E->hdf5.file_id, "/connectivity");

        status = h5write_dataset(dataset, H5T_NATIVE_INT, data, rank, memdims,
                                 offset, stride, count, block,
                                 0, (E->hdf5.cap == 0));

        status = H5Dclose(dataset);

        free(data);
    }
}


/****************************************************************************
 * Create and output /time dataset                                          *
 ****************************************************************************/

static herr_t h5create_time(hid_t loc_id)
{
    hid_t dataset;      /* dataset identifier */
    hid_t datatype;     /* row datatype identifier */
    hid_t dataspace;    /* memory dataspace */
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

    set_attribute_string(dataset, "FIELD_0_NAME", "step");
    set_attribute_string(dataset, "FIELD_1_NAME", "time");
    set_attribute_string(dataset, "FIELD_2_NAME", "time_step");
    set_attribute_string(dataset, "FIELD_3_NAME", "cpu");
    set_attribute_string(dataset, "FIELD_4_NAME", "cpu_step");

    set_attribute_double(dataset, "FIELD_0_FILL", 0);
    set_attribute_double(dataset, "FIELD_1_FILL", 0);
    set_attribute_double(dataset, "FIELD_2_FILL", 0);
    set_attribute_double(dataset, "FIELD_3_FILL", 0);
    set_attribute_double(dataset, "FIELD_4_FILL", 0);

    set_attribute_int(dataset, "AUTOMATIC_INDEX", 1);
    set_attribute_int(dataset, "REINDEX", 1);

    /* Release resources */
    status = H5Pclose(dcpl_id);
    status = H5Sclose(dataspace);
    status = H5Tclose(datatype);
    status = H5Dclose(dataset);

    return 0;
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



/****************************************************************************
 * Save most CitcomS input parameters, and other information, as            *
 * attributes in a group called /input                                      *
 ****************************************************************************/

void h5output_meta(struct All_variables *E)
{
    hid_t input;
    herr_t status;

    int n;
    int rank;
    hsize_t *dims;
    double *data;

    input = h5create_group(E->hdf5.file_id, "input", (size_t)0);

    status = set_attribute_int(input, "PID", E->control.PID);

    /*
     * Advection_diffusion.inventory
     */

    status = set_attribute_int(input, "ADV", E->advection.ADVECTION);
    status = set_attribute_int(input, "filter_temp", E->control.filter_temperature);

    status = set_attribute_float(input, "finetunedt", E->advection.fine_tune_dt);
    status = set_attribute_float(input, "fixed_timestep", E->advection.fixed_timestep);
    status = set_attribute_float(input, "inputdiffusivity", E->control.inputdiff);

    status = set_attribute_int(input, "adv_sub_iterations", E->advection.temp_iterations);


    /*
     * BC.inventory
     */

    status = set_attribute_int(input, "side_sbcs", E->control.side_sbcs);
    status = set_attribute_int(input, "pseudo_free_surf", E->control.pseudo_free_surf);

    status = set_attribute_int(input, "topvbc", E->mesh.topvbc);
    status = set_attribute_float(input, "topvbxval", E->control.VBXtopval);
    status = set_attribute_float(input, "topvbyval", E->control.VBYtopval);


    status = set_attribute_int(input, "botvbc", E->mesh.botvbc);
    status = set_attribute_float(input, "botvbxval", E->control.VBXbotval);
    status = set_attribute_float(input, "botvbyval", E->control.VBYbotval);

    status = set_attribute_int(input, "toptbc", E->mesh.toptbc);
    status = set_attribute_float(input, "toptbcval", E->control.TBCtopval);

    status = set_attribute_int(input, "bottbc", E->mesh.bottbc);
    status = set_attribute_float(input, "bottbcval", E->control.TBCbotval);

    status = set_attribute_int(input, "temperature_bound_adj", E->control.temperature_bound_adj);
    status = set_attribute_float(input, "depth_bound_adj", E->control.depth_bound_adj);
    status = set_attribute_float(input, "width_bound_adj", E->control.width_bound_adj);

    /*
     * Const.inventory
     */

    status = set_attribute_float(input, "density", E->data.density);
    status = set_attribute_float(input, "thermdiff", E->data.therm_diff);
    status = set_attribute_float(input, "gravacc", E->data.grav_acc);
    status = set_attribute_float(input, "thermexp", E->data.therm_exp);
    status = set_attribute_float(input, "refvisc", E->data.ref_viscosity);
    status = set_attribute_float(input, "cp", E->data.Cp);
    status = set_attribute_float(input, "wdensity", E->data.density_above);
    status = set_attribute_float(input, "surftemp", E->data.surf_temp);

    status = set_attribute_float(input, "z_lith", E->viscosity.zlith);
    status = set_attribute_float(input, "z_410", E->viscosity.z410);
    status = set_attribute_float(input, "z_lmantle", E->viscosity.zlm);
    status = set_attribute_float(input, "z_cmb", E->viscosity.zcmb);

    status = set_attribute_float(input, "layer_km", E->data.layer_km);
    status = set_attribute_float(input, "radius_km", E->data.radius_km);
    status = set_attribute_float(input, "scalev", E->data.scalev);
    status = set_attribute_float(input, "scalet", E->data.scalet);

    /*
     * IC.inventory
     */

    status = set_attribute_int(input, "restart", E->control.restart);
    status = set_attribute_int(input, "post_p", E->control.post_p);
    status = set_attribute_int(input, "solution_cycles_init", E->monitor.solution_cycles_init);
    status = set_attribute_int(input, "zero_elapsed_time", E->control.zero_elapsed_time);

    status = set_attribute_int(input, "tic_method", E->convection.tic_method);

    if (E->convection.tic_method == 0)
    {
        n = E->convection.number_of_perturbations;
        status = set_attribute_int(input, "num_perturbations", n);
        status = set_attribute_int_vector(input, "perturbl", n, E->convection.perturb_ll);
        status = set_attribute_int_vector(input, "perturbm", n, E->convection.perturb_mm);
        status = set_attribute_int_vector(input, "perturblayer", n, E->convection.load_depth);
        status = set_attribute_float_vector(input, "perturbmag", n, E->convection.perturb_mag);
    }
    else if (E->convection.tic_method == 1)
    {
        status = set_attribute_float(input, "half_space_age", E->convection.half_space_age);
    }
    else if (E->convection.tic_method == 2)
    {
        status = set_attribute_float(input, "half_space_age", E->convection.half_space_age);
        status = set_attribute_float_vector(input, "blob_center", 3, E->convection.blob_center);
        status = set_attribute_float(input, "blob_radius", E->convection.blob_radius);
        status = set_attribute_float(input, "blob_dT", E->convection.blob_dT);
    }

    /*
     * Param.inventory
     */

    status = set_attribute_int(input, "file_vbcs", E->control.vbcs_file);
    status = set_attribute_string(input, "vel_bound_file", E->control.velocity_boundary_file);

    status = set_attribute_int(input, "mat_control", E->control.mat_control);
    status = set_attribute_string(input, "mat_file", E->control.mat_file);

    status = set_attribute_int(input, "lith_age", E->control.lith_age);
    status = set_attribute_string(input, "lith_age_file", E->control.lith_age_file);
    status = set_attribute_int(input, "lith_age_time", E->control.lith_age_time);
    status = set_attribute_float(input, "lith_age_depth", E->control.lith_age_depth);
    status = set_attribute_float(input, "mantle_temp", E->control.lith_age_mantle_temp);

    status = set_attribute_float(input, "start_age", E->control.start_age);
    status = set_attribute_int(input, "reset_startage", E->control.reset_startage);

    /*
     * Phase.inventory
     */

    status = set_attribute_float(input, "Ra_410", E->control.Ra_410);
    status = set_attribute_float(input, "clapeyron410", E->control.clapeyron410);
    status = set_attribute_float(input, "transT410", E->control.transT410);
    status = set_attribute_float(input, "width410", E->control.width410);

    status = set_attribute_float(input, "Ra_670", E->control.Ra_670);
    status = set_attribute_float(input, "clapeyron670", E->control.clapeyron670);
    status = set_attribute_float(input, "transT670", E->control.transT670);
    status = set_attribute_float(input, "width670", E->control.width670);

    status = set_attribute_float(input, "Ra_cmb", E->control.Ra_cmb);
    status = set_attribute_float(input, "clapeyroncmb", E->control.clapeyroncmb);
    status = set_attribute_float(input, "transTcmb", E->control.transTcmb);
    status = set_attribute_float(input, "widthcmb", E->control.widthcmb);

    /*
     * Solver.inventory
     */

    status = set_attribute_string(input, "datadir", E->control.data_dir);
    status = set_attribute_string(input, "datafile", E->control.data_file);
    status = set_attribute_string(input, "datadir_old", E->control.data_dir_old);
    status = set_attribute_string(input, "datafile_old", E->control.old_P_file);

    status = set_attribute_float(input, "rayleigh", E->control.Atemp);
    status = set_attribute_float(input, "Q0", E->control.Q0);

    status = set_attribute_int(input, "stokes_flow_only", E->control.stokes);

    status = set_attribute_string(input, "output_format", E->output.format);
    status = set_attribute_string(input, "output_optional", E->output.optional);
    status = set_attribute_int(input, "verbose", E->control.verbose);
    status = set_attribute_int(input, "see_convergence", E->control.print_convergence);

    /*
     * Sphere.inventory
     */

    status = set_attribute_int(input, "nproc_surf", E->parallel.nprocxy);

    status = set_attribute_int(input, "nprocx", E->parallel.nprocx);
    status = set_attribute_int(input, "nprocy", E->parallel.nprocy);
    status = set_attribute_int(input, "nprocz", E->parallel.nprocz);

    status = set_attribute_int(input, "coor", E->control.coor);
    status = set_attribute_string(input, "coor_file", E->control.coor_file);

    status = set_attribute_int(input, "nodex", E->mesh.nox);
    status = set_attribute_int(input, "nodey", E->mesh.noy);
    status = set_attribute_int(input, "nodez", E->mesh.noz);

    status = set_attribute_int(input, "levels", E->mesh.levels);
    status = set_attribute_int(input, "mgunitx", E->mesh.mgunitx);
    status = set_attribute_int(input, "mgunity", E->mesh.mgunity);
    status = set_attribute_int(input, "mgunitz", E->mesh.mgunitz);

    status = set_attribute_double(input, "radius_outer", E->sphere.ro);
    status = set_attribute_double(input, "radius_inner", E->sphere.ri);

    status = set_attribute_int(input, "caps", E->sphere.caps);

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
        status = set_attribute_double(input, "theta_min", E->control.theta_min);
        status = set_attribute_double(input, "theta_max", E->control.theta_max);
        status = set_attribute_double(input, "fi_min", E->control.fi_min);
        status = set_attribute_double(input, "fi_max", E->control.fi_max);
    }

    status = set_attribute_int(input, "ll_max", E->sphere.llmax);
    status = set_attribute_int(input, "nlong", E->sphere.noy);
    status = set_attribute_int(input, "nlati", E->sphere.nox);
    status = set_attribute_int(input, "output_ll_max", E->sphere.output_llmax);

    /*
     * Tracer.inventory
     */

    status = set_attribute_int(input, "tracer", E->control.tracer);
    status = set_attribute_string(input, "tracer_file", E->control.tracer_file);

    /*
     * Visc.inventory
     */

    status = set_attribute_string(input, "Viscosity", E->viscosity.STRUCTURE);
    status = set_attribute_int(input, "visc_smooth_method", E->viscosity.smooth_cycles);
    status = set_attribute_int(input, "VISC_UPDATE", E->viscosity.update_allowed);

    n = E->viscosity.num_mat;
    status = set_attribute_int(input, "num_mat", n);
    status = set_attribute_float_vector(input, "visc0", n, E->viscosity.N0);
    status = set_attribute_int(input, "TDEPV", E->viscosity.TDEPV);
    status = set_attribute_int(input, "rheol", E->viscosity.RHEOL);
    status = set_attribute_float_vector(input, "viscE", n, E->viscosity.E);
    status = set_attribute_float_vector(input, "viscT", n, E->viscosity.T);
    status = set_attribute_float_vector(input, "viscZ", n, E->viscosity.Z);

    status = set_attribute_int(input, "SDEPV", E->viscosity.SDEPV);
    status = set_attribute_float(input, "sdepv_misfit", E->viscosity.sdepv_misfit);
    status = set_attribute_float_vector(input, "sdepv_expt", n, E->viscosity.sdepv_expt);

    status = set_attribute_int(input, "VMIN", E->viscosity.MIN);
    status = set_attribute_float(input, "visc_min", E->viscosity.min_value);

    status = set_attribute_int(input, "VMAX", E->viscosity.MAX);
    status = set_attribute_float(input, "visc_max", E->viscosity.max_value);

    /*
     * Incompressible.inventory
     */

    status = set_attribute_string(input, "Solver", E->control.SOLVER_TYPE);
    status = set_attribute_int(input, "node_assemble", E->control.NASSEMBLE);
    status = set_attribute_int(input, "precond", E->control.precondition);

    status = set_attribute_double(input, "accuracy", E->control.accuracy);
    status = set_attribute_float(input, "tole_compressibility", E->control.tole_comp);

    status = set_attribute_int(input, "mg_cycle", E->control.mg_cycle);
    status = set_attribute_int(input, "down_heavy", E->control.down_heavy);
    status = set_attribute_int(input, "up_heavy", E->control.up_heavy);

    status = set_attribute_int(input, "vlowstep", E->control.v_steps_low);
    status = set_attribute_int(input, "vhighstep", E->control.v_steps_high);
    status = set_attribute_int(input, "piterations", E->control.p_iterations);

    status = set_attribute_int(input, "aug_lagr", E->control.augmented_Lagr);
    status = set_attribute_double(input, "aug_number", E->control.augmented);

    /* status = set_attribute(input, "", H5T_NATIVE_, &(E->)); */

    /*
     * Release resources
     */
    status = H5Gclose(input);
}

/****************************************************************************
 * Some of the following functions were based from the H5ATTR.c             *
 * source file in PyTables, which is a BSD-licensed python extension        *
 * for accessing HDF5 files.                                                *
 *                                                                          *
 * The copyright notice is hereby retained.                                 *
 *                                                                          *
 * NCSA HDF                                                                 *
 * Scientific Data Technologies                                             *
 * National Center for Supercomputing Applications                          *
 * University of Illinois at Urbana-Champaign                               *
 * 605 E. Springfield, Champaign IL 61820                                   *
 *                                                                          *
 * For conditions of distribution and use, see the accompanying             *
 * hdf/COPYING file.                                                        *
 *                                                                          *
 * Modified versions of H5LT for getting and setting attributes for open    *
 * groups and leaves.                                                       *
 * F. Altet 2005/09/29                                                      *
 *                                                                          *
 ****************************************************************************/

/* Function  : find_attr
 * Purpose   : operator function used by find_attribute
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 * Date      : June 21, 2001
 */
static herr_t find_attr(hid_t loc_id, const char *name, void *op_data)
{
    /* Define a default zero value for return. This will cause the
     * iterator to continue if the palette attribute is not found yet.
     */

    int ret = 0;

    char *attr_name = (char *)op_data;

    /* Shut the compiler up */
    loc_id = loc_id;

    /* Define a positive value for return value if the attribute was
     * found. This will cause the iterator to immediately return that
     * positive value, indicating short-circuit success
     */

    if(strcmp(name, attr_name) == 0)
        ret = 1;

    return ret;
}

/* Function  : find_attribute
 * Purpose   : Inquires if an attribute named attr_name exists attached
 *             attached to the object loc_id.
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 * Date      : June 21, 2001
 *
 * Comments:
 *  The function uses H5Aiterate with the operator function find_attr
 *
 * Return:
 *
 *  Success: The return value of the first operator that returns
 *           non-zero, or zero if all members were processed with no
 *           operator returning non-zero.
 *
 *  Failure: Negative if something goes wrong within the library,
 *           or the negative value returned by one of the operators.
 */
static herr_t find_attribute(hid_t loc_id, const char *attr_name)
{
    unsigned int attr_num;
    herr_t ret;

    attr_num = 0;
    ret = H5Aiterate(loc_id, &attr_num, find_attr, (void *)attr_name);

    return ret;
}


/* Function: set_attribute_string
 * Purpose : Creates and writes a string attribute named attr_name
 *           and attaches it to the object specified by obj_id
 * Return  : Success 0, Failure -1
 * Comments: If the attribute already exists, it is overwritten.
 */
herr_t set_attribute_string(hid_t obj_id,
                            const char *attr_name,
                            const char *attr_data)
{
    hid_t   attr_type;
    hid_t   attr_size;
    hid_t   attr_space_id;
    hid_t   attr_id;
    int     has_attr;
    herr_t  status;

    /* Create the attribute */
    attr_type = H5Tcopy(H5T_C_S1);
    if (attr_type < 0) goto out;

    attr_size = strlen(attr_data) + 1;  /* extra null term */

    status = H5Tset_size(attr_type, (size_t)attr_size);
    if (status < 0) goto out;

    status = H5Tset_strpad(attr_type, H5T_STR_NULLTERM);
    if (status < 0) goto out;

    attr_space_id = H5Screate(H5S_SCALAR);
    if (status < 0) goto out;

    /* Verify if the attribute already exists */
    has_attr = find_attribute(obj_id, attr_name);

    /* The attribute already exists, delete it */
    if (has_attr == 1)
    {
        status = H5Adelete(obj_id, attr_name);
        if (status < 0) goto out;
    }

    /* Create and write the attribute */

    attr_id = H5Acreate(obj_id, attr_name, attr_type, attr_space_id,
                        H5P_DEFAULT);
    if(attr_id < 0) goto out;

    status = H5Awrite(attr_id, attr_type, attr_data);
    if(status < 0) goto out;

    status = H5Aclose(attr_id);
    if(status < 0) goto out;

    status = H5Sclose(attr_space_id);
    if(status < 0) goto out;

    status = H5Tclose(attr_type);
    if(status < 0) goto out;


    return 0;

out:
    return -1;
}


/* Function  : set_attribute
 * Purpose   : Private function used by
 *             set_attribute_int and set_attribute_float
 * Return    : Success 0, Failure -1
 * Programmer: Pedro Vicente, pvn@ncsa.uiuc.edu
 * Date      : July 25, 2001
 */
herr_t set_attribute(hid_t obj_id,
                     const char *attr_name,
                     hid_t type_id,
                     const void *data)
{
    hid_t space_id, attr_id;
    herr_t status;

    int has_attr;

    /* Create the data space for the attribute. */
    space_id = H5Screate(H5S_SCALAR);
    if (space_id < 0) goto out;

    /* Verify if the attribute already exists */
    has_attr = find_attribute(obj_id, attr_name);
    if (has_attr == 1)
    {
        /* The attribute already exists. Delete it. */
        status = H5Adelete(obj_id, attr_name);
        if(status < 0) goto out;
    }

    /* Create the attribute. */
    attr_id = H5Acreate(obj_id, attr_name, type_id, space_id, H5P_DEFAULT);
    if (attr_id < 0) goto out;

    /* Write the attribute data. */
    status = H5Awrite(attr_id, type_id, data);
    if (status < 0) goto out;

    /* Close the attribute. */
    status = H5Aclose(attr_id);
    if (status < 0) goto out;

    /* Close the data space. */
    status = H5Sclose(space_id);
    if (status < 0) goto out;

    return 0;

out:
    return -1;
}

herr_t set_attribute_float(hid_t obj_id, const char *attr_name, float x)
{
    return set_attribute(obj_id, attr_name, H5T_NATIVE_FLOAT, &x);
}

herr_t set_attribute_double(hid_t obj_id, const char *attr_name, double x)
{
    return set_attribute(obj_id, attr_name, H5T_NATIVE_DOUBLE, &x);
}

herr_t set_attribute_int(hid_t obj_id, const char *attr_name, int n)
{
    return set_attribute(obj_id, attr_name, H5T_NATIVE_INT, &n);
}

herr_t set_attribute_long(hid_t obj_id, const char *attr_name, long n)
{
    return set_attribute(obj_id, attr_name, H5T_NATIVE_LONG, &n);
}

herr_t set_attribute_llong(hid_t obj_id, const char *attr_name, long long n)
{
    return set_attribute(obj_id, attr_name, H5T_NATIVE_LLONG, &n);
}

/* Function: set_attribute_array
 * Purpose : write an array attribute
 * Return  : Success 0, Failure -1
 * Date    : July 25, 2001
 */
herr_t set_attribute_array(hid_t obj_id,
                           const char *attr_name,
                           size_t rank,
                           hsize_t *dims,
                           hid_t type_id,
                           const void *data)
{
    hid_t space_id, attr_id;
    herr_t status;

    int has_attr;

    /* Create the data space for the attribute. */
    space_id = H5Screate_simple(rank, dims, NULL);
    if (space_id < 0) goto out;

    /* Verify if the attribute already exists. */
    has_attr = find_attribute(obj_id, attr_name);
    if (has_attr == 1)
    {
        /* The attribute already exists. Delete it. */
        status = H5Adelete(obj_id, attr_name);
        if (status < 0) goto out;
    }

    /* Create the attribute. */
    attr_id = H5Acreate(obj_id, attr_name, type_id, space_id, H5P_DEFAULT);
    if (attr_id < 0) goto out;

    /* Write the attribute data. */
    status = H5Awrite(attr_id, type_id, data);
    if (status < 0) goto out;

    /* Close the attribute. */
    status = H5Aclose(attr_id);
    if (status < 0) goto out;

    /* Close the dataspace. */
    status = H5Sclose(space_id);
    if (status < 0) goto out;

    return 0;

out:
    return -1;
}

herr_t set_attribute_vector(hid_t obj_id,
                            const char *attr_name,
                            hsize_t dim,
                            hid_t type_id,
                            const void *data)
{
    return set_attribute_array(obj_id, attr_name, 1, &dim, type_id, data);
}

herr_t set_attribute_int_vector(hid_t obj_id,
                                const char *attr_name,
                                hsize_t dim,
                                const int *data)
{
    return set_attribute_array(obj_id, attr_name, 1, &dim, H5T_NATIVE_INT, data);
}

herr_t set_attribute_float_vector(hid_t obj_id,
                                  const char *attr_name,
                                  hsize_t dim,
                                  const float *data)
{
    return set_attribute_array(obj_id, attr_name, 1, &dim, H5T_NATIVE_FLOAT, data);
}

herr_t set_attribute_double_vector(hid_t obj_id,
                                   const char *attr_name,
                                   hsize_t dim,
                                   const double *data)
{
    return set_attribute_array(obj_id, attr_name, 1, &dim, H5T_NATIVE_DOUBLE, data);
}

#endif  /* #ifdef USE_HDF5 */
