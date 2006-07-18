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

/* TODO: include additional copyright notices from Pytables.c in here */

#if !defined(CitcomS_pytables_h)
#define CitcomS_pytables_h

#ifdef __cplusplus
extern "C" {
#endif

/* hardcoded pickle of Filter class */
#define FILTERS_P "ccopy_reg\n_reconstructor\np1\n(ctables.Leaf\nFilters\np2\nc__builtin__\nobject\np3\nNtRp4\n(dp5\nS'shuffle'\np6\nI0\nsS'complevel'\np7\nI0\nsS'fletcher32'\np8\nI0\nsS'complib'\np9\nS'zlib'\np10\nsb."


herr_t find_attribute(hid_t loc_id, const char *attr_name);

herr_t set_attribute_string(hid_t obj_id,
                            const char *attr_name,
                            const char *attr_data);

herr_t make_array(hid_t loc_id,
                  const char *dset_name,
                  const int rank,
                  const hsize_t *dims,
                  hid_t type_id,
                  const void *data);



#ifdef __cplusplus
}
#endif

#endif
