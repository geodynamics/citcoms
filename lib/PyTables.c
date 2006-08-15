/* PyTables.c - This file exposes an API to create HDF5 files that
 *              are compatible with PyTables.
 *
 * TODO: display appropriate copyright notices, for the functions
 *       that were taken from the PyTables source.
 *
 */


#ifdef USE_HDF5

#include "hdf5.h"
#include "pytables.h"


/****************************************************************************
 * Source: H5ATTR.c                                                         *
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
herr_t find_attribute(hid_t loc_id, const char *attr_name)
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


#endif  /* #ifdef USE_HDF5 */
