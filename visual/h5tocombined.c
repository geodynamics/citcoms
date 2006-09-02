/*
 * h5tocombined.c by Luis Armendariz.
 * Copyright (C) 2006, California Institute of Technology.
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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "hdf5.h"


typedef struct cap_t
{
    int id;
    char name[8];
    hid_t group;
} cap_t;


typedef struct field_t
{
    const char *name;
    
    int rank;
    hsize_t *dims;
    hsize_t *maxdims;

    hsize_t *offset;
    hsize_t *count;

    int n;
    float *data;

} field_t;


static cap_t *open_cap(hid_t file_id, int capid);
static herr_t close_cap(cap_t *cap);


static field_t *open_field(cap_t *cap, const char *name);
static herr_t read_field(cap_t *cap, field_t *field, int timestep);
static herr_t close_field(field_t *field);


static herr_t get_attribute_str(hid_t obj_id, const char *attr_name, char **data);
static herr_t get_attribute_int(hid_t input, const char *name, int *val);
static herr_t get_attribute(hid_t obj_id, const char *attr_name, hid_t mem_type_id, void *data);
static herr_t get_attribute_mem(hid_t obj_id, const char *attr_name, hid_t mem_type_id, void *data);
static herr_t get_attribute_disk(hid_t loc_id, const char *attr_name, void *attr_out);
static herr_t get_attribute_info(hid_t obj_id, const char *attr_name, hsize_t *dims, H5T_class_t *type_class, size_t *type_size, hid_t *type_id);


int main(int argc, char *argv[])
{
    FILE *file;
    char filename[100];
    char *datafile;

    hid_t h5file;
    hid_t input;
    herr_t status;

    int id;
    int caps;
    cap_t **cap;

    int step;
    int n, i, j, k;
    int nodex, nodey, nodez;

    int t;
    int timesteps;

    int *steps;
    char *endptr;

    field_t *coord;
    field_t *velocity;
    field_t *temperature;
    field_t *viscosity;


    /************************************************************************
     * Parse command-line parameters.                                       *
     ************************************************************************/

    /*
     * HDF5 file must be specified as first argument.
     */
    
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s file.h5 [step1 [step2 [...]]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    /*
     * Recognize help arguments.
     */

    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
    {
        fprintf(stderr, "Usage: %s file.h5 [step1 [step2 [...]]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    /*
     * Open HDF5 file (read-only). Complain if invalid.
     */

    h5file = H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5file < 0)
    {
        fprintf(stderr, "Could not open HDF5 file \"%s\"\n", argv[1]);
        return EXIT_FAILURE;
    }

    /*
     * Read step(s) from argv[2:]
     */
    
    /* Allocate at least one step (we know argc >= 2) */
    timesteps = (argc == 2) ? 1 : (argc-2);
    steps = (int *)malloc(timesteps * sizeof(int));

    /* By default, use zeroth step (might be overwritten) */
    steps[0] = 0;

    /* Convert argv[2:] into int array */
    for(n = 2; n < argc; n++)
    {
        steps[n-2] = (int)strtol(argv[n], &endptr, 10);
        if (!(argv[n][0] != '\0' && *endptr == '\0'))
        {
            fprintf(stderr, "Error: Could not parse step \"%s\"\n", argv[n]);
            return EXIT_FAILURE;
        }
    }


    /************************************************************************
     * Get mesh parameters.                                                 *
     ************************************************************************/

    /* Read input group */
    input = H5Gopen(h5file, "input");
    if (input < 0)
    {
        fprintf(stderr, "Could not open /input group in \"%s\"\n", argv[1]);
        status = H5Fclose(h5file);
        return EXIT_FAILURE;
    }

    status = get_attribute_str(input, "datafile", &datafile);
    status = get_attribute_int(input, "nproc_surf", &caps);
    status = get_attribute_int(input, "nodex", &nodex);
    status = get_attribute_int(input, "nodey", &nodey);
    status = get_attribute_int(input, "nodez", &nodez);

    /* Release input group */
    status = H5Gclose(input);


    /************************************************************************
     * Open all available cap groups.                                       *
     ************************************************************************/

    cap = (cap_t **)malloc(caps * sizeof(cap_t *));

    for(id = 0; id < caps; id++)
        cap[id] = open_cap(h5file, id);


    /************************************************************************
     * Create fields using cap00 datasets as a template.                    *
     ************************************************************************/

    coord       = open_field(cap[0], "coord");
    velocity    = open_field(cap[0], "velocity");
    temperature = open_field(cap[0], "temperature");
    viscosity   = open_field(cap[0], "viscosity");


    /************************************************************************
     * Output requested data.                                               *
     ************************************************************************/

    /* Iterate over timesteps */
    for(t = 0; t < timesteps; t++)
    {
        /* Determine step */
        step = steps[t];

        /* Iterate over caps */
        for(id = 0; id < caps; id++)
        {
            snprintf(filename, (size_t)99, "%s.cap%02d.%d", datafile, id, step);
            fprintf(stderr, "Writing %s\n", filename);

            file = fopen(filename, "w");
            fprintf(file, "%d x %d x %d\n", nodex, nodey, nodez);

            /* Read data from HDF5 file. */
            read_field(cap[id], coord, 0);
            read_field(cap[id], velocity, step);
            read_field(cap[id], temperature, step);
            read_field(cap[id], viscosity, step);

            /* Traverse data in Citcom order */
            n = 0;
            for(j = 0; j < nodey; j++)
            {
                for(i = 0; i < nodex; i++)
                {
                    for(k = 0; k < nodez; k++)
                    {
                        fprintf(file, "%g %g %g %g %g %g %g %g\n",
                                coord->data[3*n+0],
                                coord->data[3*n+1],
                                coord->data[3*n+2],
                                velocity->data[3*n+0],
                                velocity->data[3*n+1],
                                velocity->data[3*n+2],
                                temperature->data[n],
                                viscosity->data[n]);

                        n++;    /* n = k + i*nodez + j*nodez*nodex */
                    }
                }
            }

            fclose(file);
        }
    }

    /* Release resources. */

    for(id = 0; id < caps; id++)
        status = close_cap(cap[id]);
    free(cap);

    status = close_field(coord);
    status = close_field(velocity);
    status = close_field(temperature);
    status = close_field(viscosity);
    status = H5Fclose(h5file);

    free(steps);

    return EXIT_SUCCESS;
}


static cap_t *open_cap(hid_t file_id, int capid)
{
    cap_t *cap;
    cap = (cap_t *)malloc(sizeof(cap_t));
    cap->id = capid;
    snprintf(cap->name, (size_t)7, "cap%02d", capid);
    cap->group = H5Gopen(file_id, cap->name);
    if (cap->group < 0)
    {
        free(cap);
        return NULL;
    }
    return cap;
}


static herr_t close_cap(cap_t *cap)
{
    herr_t status;
    if (cap != NULL)
    {
        cap->id = -1;
        cap->name[0] = '\0';
        status = H5Gclose(cap->group);
        free(cap);
    }
    return 0;
}


static field_t *open_field(cap_t *cap, const char *name)
{
    hid_t dataset;
    hid_t dataspace;
    herr_t status;

    int d;
    int rank;

    field_t *field;

    if (cap == NULL)
        return NULL;


    /* Allocate field and initialize. */

    field = (field_t *)malloc(sizeof(field_t));

    field->name = name;
    field->rank = 0;
    field->dims = NULL;
    field->maxdims = NULL;
    field->n = 0;

    dataset = H5Dopen(cap->group, name);
    if(dataset < 0)
    {
        free(field);
        return NULL;
    }

    dataspace = H5Dget_space(dataset);
    if (dataspace < 0)
    {
        free(field);
        return NULL;
    }


    /* Calculate shape of field. */

    rank = H5Sget_simple_extent_ndims(dataspace);

    field->rank = rank;
    field->dims = (hsize_t *)malloc(rank * sizeof(hsize_t));
    field->maxdims = (hsize_t *)malloc(rank * sizeof(hsize_t));

    status = H5Sget_simple_extent_dims(dataspace, field->dims, field->maxdims);

    /* DEBUG
    printf("Field %s shape (", name);
    for(d = 0; d < rank; d++)
        printf("%d,", (int)(field->dims[d]));
    printf(")\n");
    // */


    /* Allocate memory for hyperslab selection parameters. */

    field->offset = (hsize_t *)malloc(rank * sizeof(hsize_t));
    field->count  = (hsize_t *)malloc(rank * sizeof(hsize_t));


    /* Allocate enough memory for a single time-slice buffer. */

    field->n = 1;
    if (field->maxdims[0] == H5S_UNLIMITED)
        for(d = 1; d < rank; d++)
            field->n *= field->dims[d];
    else
        for(d = 0; d < rank; d++)
            field->n *= field->dims[d];

    field->data = (float *)malloc(field->n * sizeof(float));


    /* Release resources. */

    status = H5Sclose(dataspace);
    status = H5Dclose(dataset);

    return field;
}


static herr_t read_field(cap_t *cap, field_t *field, int timestep)
{
    hid_t dataset;
    hid_t filespace;
    hid_t memspace;
    herr_t status;

    int d;

    if (cap == NULL || field == NULL)
        return -1;

    dataset = H5Dopen(cap->group, field->name);

    if (dataset < 0)
        return -1;

    for(d = 0; d < field->rank; d++)
    {
        field->offset[d] = 0;
        field->count[d]  = field->dims[d];
    }

    if (field->maxdims[0] == H5S_UNLIMITED)
    {
        field->offset[0] = timestep;
        field->count[0]  = 1;
    }

    /* DEBUG
    printf("Reading step %d on field %s with offset (", timestep, field->name);
    for(d = 0; d < field->rank; d++) printf("%d,", (int)(field->offset[d]));
    printf(") and count (");
    for(d = 0; d < field->rank; d++) printf("%d,", (int)(field->count[d]));
    printf(")\n");
    // */


    filespace = H5Dget_space(dataset);

    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                 field->offset, NULL, field->count, NULL);

    memspace = H5Screate_simple(field->rank, field->count, NULL);

    status = H5Dread(dataset, H5T_NATIVE_FLOAT, memspace,
                     filespace, H5P_DEFAULT, field->data);
    
    status = H5Sclose(filespace);
    status = H5Sclose(memspace);
    status = H5Dclose(dataset);

    return 0;
}


static herr_t close_field(field_t *field)
{
    if (field != NULL)
    {
        free(field->dims);
        free(field->maxdims);
        free(field->offset);
        free(field->count);
        free(field->data);
        free(field);
    }
    return 0;
}


static herr_t get_attribute_str(hid_t obj_id,
                                const char *attr_name,
                                char **data)
{
    hid_t attr_id;
    hid_t attr_type;
    size_t type_size;
    herr_t status;

    *data = NULL;

    attr_id = H5Aopen_name(obj_id, attr_name);
    if (attr_id < 0)
        return -1;

    attr_type = H5Aget_type(attr_id);
    if (attr_type < 0)
        goto out;

    /* Get the size */
    type_size = H5Tget_size(attr_type);
    if (type_size < 0)
        goto out;

    /* malloc enough space for the string, plus 1 for trailing '\0' */
    *data = (char *)malloc(type_size + 1);

    status = H5Aread(attr_id, attr_type, *data);
    if (status < 0)
        goto out;

    /* Set the last character to '\0' in case we are dealing with
     * null padded or space padded strings
     */
    (*data)[type_size] = '\0';

    status = H5Tclose(attr_type);
    if (status < 0)
        goto out;

    status = H5Aclose(attr_id);
    if (status < 0)
        return -1;

    return 0;

out:
    H5Tclose(attr_type);
    H5Aclose(attr_id);
    if (*data)
        free(*data);
    return -1;
}


static herr_t get_attribute_int(hid_t input, const char *name, int *val)
{
    hid_t attr_id;
    hid_t type_id;
    H5T_class_t type_class;
    size_t type_size;

    herr_t status;

    char *strval;

    attr_id = H5Aopen_name(input, name);
    type_id = H5Aget_type(attr_id);
    type_class = H5Tget_class(type_id);
    type_size = H5Tget_size(type_id);

    H5Tclose(type_id);
    H5Aclose(attr_id);

    switch(type_class)
    {
        case H5T_STRING:
            status = get_attribute_str(input, name, &strval);
            if (status < 0) return -1;
            *val = atoi(strval);
            free(strval);
            return 0;
        case H5T_INTEGER:
            status = get_attribute(input, name, H5T_NATIVE_INT, val);
            if (status < 0) return -1;
            return 0;
    }

    return -1;
}


static herr_t get_attribute(hid_t obj_id,
                            const char *attr_name,
                            hid_t mem_type_id,
                            void *data)
{
    herr_t status;

    status = get_attribute_mem(obj_id, attr_name, mem_type_id, data);
    if (status < 0)
        return -1;

    return 0;
}


static herr_t get_attribute_mem(hid_t obj_id,
                                const char *attr_name,
                                hid_t mem_type_id,
                                void *data)
{
    hid_t attr_id;
    herr_t status;

    attr_id = H5Aopen_name(obj_id, attr_name);
    if (attr_id < 0)
        return -1;

    status = H5Aread(attr_id, mem_type_id, data);
    if (status < 0)
    {
        H5Aclose(attr_id);
        return -1;
    }

    status = H5Aclose(attr_id);
    if (status < 0)
        return -1;

    return 0;
}


static herr_t get_attribute_disk(hid_t loc_id,
                                 const char *attr_name,
                                 void *attr_out)
{
    hid_t attr_id;
    hid_t attr_type;
    herr_t status;

    attr_id = H5Aopen_name(loc_id, attr_name);
    if (attr_id < 0)
        return -1;

    attr_type = H5Aget_type(attr_id);
    if (attr_type < 0)
        goto out;

    status = H5Aread(attr_id, attr_type, attr_out);
    if (status < 0)
        goto out;

    status = H5Tclose(attr_type);
    if (status < 0)
        goto out;

    status = H5Aclose(attr_id);
    if (status < 0)
        return -1;

    return 0;
out:
    H5Tclose(attr_type);
    H5Aclose(attr_id);
    return -1;
}


static herr_t get_attribute_info(hid_t obj_id,
                                 const char *attr_name,
                                 hsize_t *dims,
                                 H5T_class_t *type_class,
                                 size_t *type_size,
                                 hid_t *type_id)
{
    hid_t attr_id;
    hid_t space_id;
    herr_t status;
    int rank;

    /* Open the attribute. */
    attr_id = H5Aopen_name(obj_id, attr_name);
    if (attr_id < 0)
        return -1;

    /* Get an identifier for the datatype. */
    *type_id = H5Aget_type(attr_id);

    /* Get the class. */
    *type_class = H5Tget_class(*type_id);

    /* Get the size. */
    *type_size = H5Tget_size(*type_id);

    /* Get the dataspace handle */
    space_id = H5Aget_space(attr_id);
    if (space_id < 0)
        goto out;

    /* Get dimensions */
    rank = H5Sget_simple_extent_dims(space_id, dims, NULL);
    if (rank < 0)
        goto out;

    /* Terminate access to the dataspace */
    status = H5Sclose(space_id);
    if (status < 0)
        goto out;

    /* End access to the attribute */
    status = H5Aclose(attr_id);
    if (status < 0)
        goto out;

    return 0;
out:
    H5Tclose(*type_id);
    H5Aclose(attr_id);
    return -1;
}

