/*
 * h5util.h by Luis Armendariz and Eh Tan.
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


#include "hdf5.h"


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


field_t *open_field(hid_t cap, const char *name);
herr_t read_field(hid_t cap, field_t *field, int iii);
herr_t close_field(field_t *field);

herr_t get_attribute_str(hid_t obj_id, const char *attr_name, char **data);
herr_t get_attribute_int(hid_t input, const char *name, int *val);
herr_t get_attribute_float(hid_t input, const char *name, float *val);

