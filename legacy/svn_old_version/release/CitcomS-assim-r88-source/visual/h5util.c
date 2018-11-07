/*
 * h5util.c by Luis Armendariz and Eh Tan.
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
#include "h5util.h"


static herr_t get_attribute(hid_t obj_id, const char *attr_name, hid_t mem_type_id, void *data);
static herr_t get_attribute_mem(hid_t obj_id, const char *attr_name, hid_t mem_type_id, void *data);


field_t *open_field(hid_t group, const char *name)
{
    hid_t dataset;
    hid_t dataspace;
    herr_t status;

    int d;
    int rank;

    field_t *field;

    if (group < 0)
        return NULL;


    /* Allocate field and initialize. */

    field = (field_t *)malloc(sizeof(field_t));

    field->name = name;
    field->rank = 0;
    field->dims = NULL;
    field->maxdims = NULL;
    field->n = 0;

    dataset = H5Dopen(group, name);
    if(dataset < 0)
    {
        free(field);
        fprintf(stderr, "Could not open HDF5 dataset \"%s\"\n", name);
        return NULL;
    }

    dataspace = H5Dget_space(dataset);
    if (dataspace < 0)
    {
        free(field);
        fprintf(stderr, "Could not open HDF5 dataspace \"%s\"\n", name);
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


    /* Allocate enough memory for a single cap-slice buffer. */

    field->n = 1;
    for(d = 1; d < rank; d++)
	field->n *= field->dims[d];

    field->data = (float *)malloc(field->n * sizeof(float));


    /* Release resources. */

    status = H5Sclose(dataspace);
    status = H5Dclose(dataset);

    return field;
}


herr_t read_field(hid_t group, field_t *field, int cap)
{
    hid_t dataset;
    hid_t filespace;
    hid_t memspace;
    herr_t status;

    int d;

    if (group < 0 || field == NULL)
        return -1;

    dataset = H5Dopen(group, field->name);

    if (dataset < 0)
        return -1;

    field->offset[0] = cap;
    field->count[0] = 1;
    for(d = 1; d < field->rank; d++)
    {
        field->offset[d] = 0;
        field->count[d]  = field->dims[d];
    }


    /* DEBUG
    printf("Reading cap %d on field %s with offset (", cap, field->name);
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


herr_t close_field(field_t *field)
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

herr_t get_attribute_str(hid_t obj_id,
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

herr_t get_attribute_int(hid_t input, const char *name, int *val)
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
        default:
	    return -1;
    }

    return -1;
}


herr_t get_attribute_float(hid_t input, const char *name, float *val)
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
            *val = atof(strval);
            free(strval);
            return 0;
        case H5T_FLOAT:
            status = get_attribute(input, name, H5T_NATIVE_FLOAT, val);
            if (status < 0) return -1;
            return 0;
        default:
	    return -1;
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


