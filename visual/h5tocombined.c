/*
 * CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
 * Copyright (C) 2002-2005, California Institute of Technology.
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
    hid_t dataset;

    int step;
    
    int rank;
    hsize_t *dims;
    hsize_t *maxdims;

    hsize_t *offset;
    hsize_t *count;

    int n;
    float *data;

} field_t;


static cap_t *open_cap(hid_t file_id, int capid)
{
    cap_t *cap;
    cap = (cap_t *)malloc(sizeof(cap_t));
    cap->id = capid;
    snprintf(cap->name, (size_t)7, "cap%02d", capid);
    cap->group = H5Gopen(file_id, cap->name);
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


static field_t *open_field(hid_t loc_id, const char *name)
{
    hid_t dataspace;
    herr_t status;

    int d;
    int rank;

    field_t *field;

    /*
     * Allocate field and initialize values.
     */

    field = (field_t *)malloc(sizeof(field_t));

    field->rank = 0;
    field->dims = NULL;
    field->maxdims = NULL;

    field->n = 0;
    field->step = -1;

    field->name = name;
    field->dataset = H5Dopen(loc_id, name);
    if(field-> dataset < 0)
    {
        free(field);
        return NULL;
    }

    dataspace = H5Dget_space(field->dataset);
    if (dataspace < 0)
    {
        free(field);
        return NULL;
    }

    /*
     * Calculate shape of field.
     */

    rank = H5Sget_simple_extent_ndims(dataspace);

    field->rank = rank;
    field->dims = (hsize_t *)malloc(rank * sizeof(hsize_t));
    field->maxdims = (hsize_t *)malloc(rank * sizeof(hsize_t));

    status = H5Sget_simple_extent_dims(dataspace, field->dims, field->maxdims);

    /*
     * Allocate memory for hyperslab selection parameters.
     */

    field->offset = (hsize_t *)malloc(rank * sizeof(hsize_t));
    field->count  = (hsize_t *)malloc(rank * sizeof(hsize_t));

    /*
     * Allocate enough memory for a single time-slice.
     */

    field->n = 1;
    if (field->maxdims[0] == H5S_UNLIMITED)
        for(d = 1; d < rank; d++)
            field->n *= field->dims[d];
    else
        for(d = 0; d < rank; d++)
            field->n *= field->dims[d];

    /* DEBUG
    printf("Field %s shape (", name);
    for(d = 0; d < rank; d++)
        printf("%d,", (int)(field->dims[d]));
    printf(")\n");
    // */

    field->data = (float *)malloc(field->n * sizeof(float));

    status = H5Sclose(dataspace);
    status = H5Dclose(field->dataset);

    return field;
}

static herr_t read_field(field_t *field, int timestep)
{
    hid_t filespace;
    hid_t memspace;
    herr_t status;

    int d;

    field->step = timestep; /* which step does field->data correspond to? */

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

    filespace = H5Dget_space(field->dataset);

    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                 field->offset, NULL, field->count, NULL);

    memspace = H5Screate_simple(field->rank, field->count, NULL);

    status = H5Dread(field->dataset, H5T_NATIVE_FLOAT, memspace,
                     filespace, H5P_DEFAULT, field->data);
    
    status = H5Sclose(filespace);
    status = H5Sclose(memspace);
    status = H5Dclose(field->dataset);

    return 0;
}

static int free_field(field_t *field)
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


/* TODO: deallocating values allocated in this function? */
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

static herr_t get_attribute_string(hid_t obj_id,
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

static herr_t get_parameter_int(hid_t input, const char *name, int *val)
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
            status = get_attribute_string(input, name, &strval);
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


int main(int argc, char *argv[])
{
    FILE *file;
    char filename[100];
    char *datafile;

    hid_t h5file;
    hid_t input;
    herr_t status;

    int caps;
    int capid;
    cap_t *cap;

    int step;
    int n, i, j, k;
    int nodex, nodey, nodez;

    field_t *coord;
    field_t *velocity;
    field_t *temperature;
    field_t *viscosity;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s file.h5 step1 [step2 [...] ] \n", argv[0]);
        return EXIT_FAILURE;
    }

    /* Open HDF5 file (read-only) */
    h5file = H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5file < 0)
    {
        fprintf(stderr, "Could not open HDF5 file \"%s\"\n", argv[1]);
        return EXIT_FAILURE;
    }

    /* TODO: read step(s) from command-line. For now, use single timestep */
    step = 0;

    /* Read input group */
    input = H5Gopen(h5file, "input");
    if (input < 0)
    {
        fprintf(stderr, "Could not open /input group in \"%s\"\n", argv[1]);
        status = H5Fclose(h5file);
        return EXIT_FAILURE;
    }

    /* Read datafile from /input */
    get_attribute_string(input, "datafile", &datafile);

    /* Read number of caps from /input */
    get_parameter_int(input, "nproc_surf", &caps);

    /* Release input group */
    status = H5Gclose(input);

    /*
     * Read first cap and create fields
     */
    cap = open_cap(h5file, 0);
    
    velocity    = open_field(cap->group, "velocity");
    temperature = open_field(cap->group, "temperature");
    viscosity   = open_field(cap->group, "viscosity");

    coord = open_field(cap->group, "coord");
    coord->dataset = H5Dopen(cap->group, "coord");
    read_field(coord, 0);
    nodex = coord->dims[0];
    nodey = coord->dims[1];
    nodez = coord->dims[2];
    close_cap(cap);

    for(capid = 0; capid < caps; capid++)
    {
        cap = open_cap(h5file, capid);

        snprintf(filename, (size_t)99, "%s.cap%02d.%d", datafile, capid, step);
        fprintf(stderr, "Writing %s\n", filename);

        file = fopen(filename, "w");
        fprintf(file, "%d x %d x %d\n", nodex, nodey, nodez);

        velocity->dataset    = H5Dopen(cap->group, "velocity");
        temperature->dataset = H5Dopen(cap->group, "temperature");
        viscosity->dataset   = H5Dopen(cap->group, "viscosity");

        /*
         * Read data from HDF5 file.
         */
        read_field(velocity, step);
        read_field(temperature, step);
        read_field(viscosity, step);

        /*
         * Traverse data in order n = k + i*nodez + j*nodez*nodex
         */
        n = 0;
        for(j = 0; j < nodey; j++)
        {
            for(i = 0; i < nodex; i++)
            {
                for(k = 0; k < nodez; k++)
                {
                    fprintf(file, "%g %g %g %g %g %g %g %g\n",
                            coord->data[3*n],
                            coord->data[3*n+1],
                            coord->data[3*n+2],
                            velocity->data[3*n],
                            velocity->data[3*n+1],
                            velocity->data[3*n+2],
                            temperature->data[n],
                            viscosity->data[n]);
                    n++;
                }
            }
        }

        fclose(file);
        close_cap(cap);
    }

    free_field(coord);
    free_field(velocity);
    free_field(temperature);
    free_field(viscosity);

    status = H5Fclose(h5file);
    return EXIT_SUCCESS;

}
