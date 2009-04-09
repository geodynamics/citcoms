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

#if !defined(CitcomS_output_gzdir_h)
#define CitcomS_output_gzdir_h

#ifdef USE_GZDIR

#include <stdio.h>
#include <zlib.h>

struct All_variables;

void gzdir_output(struct All_variables *E, int out_cycles);
void restart_tic_from_gzdir_file(struct All_variables *E);
void gzip_file(char *output_file);
void get_vtk_filename(char *output_file, int geo, struct All_variables *E, int cycles);
gzFile *gzdir_output_open(char *,char *);
void restart_tic_from_gzdir_file(struct All_variables *);
int open_file_zipped(char *, FILE **,struct All_variables *);
void gzip_file(char *);

#endif

#endif
