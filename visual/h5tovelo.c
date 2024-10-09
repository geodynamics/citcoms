/*
 * h5tovelo.c by Eh Tan and Luis Armendariz.
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
#include "hdf5.h"
#include "h5util.h"


static const char usage[] =
    "Convert h5 files to velo files, for restart purpose\n"
    "\n"
    "Usage: h5tovelo modelname step\n"
    "\n"
    "modelname: prefix of the CitcomS HDF5 datafile\n"
    "step: time step\n";



int main(int argc, char *argv[])
{
    char filename1[100];
    char filename2[100];
    char prefix[100];

    hid_t h5file1, h5file2;
    hid_t input;
    herr_t status;

    int caps;
    int cap;

    int nprocx, nprocy, nprocz;
    int nodex, nodey, nodez;

    int step;
    float time;

    field_t *velocity;
    field_t *temperature;


    /************************************************************************
     * Parse command-line parameters.                                       *
     ************************************************************************/

    /*
     * HDF5 file must be specified as first argument.
     */

    if (argc < 3 || argc > 4)
    {
	fputs(usage, stderr);
        return EXIT_FAILURE;
    }

    /*
     * Recognize help arguments.
     */

    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
    {
	fputs(usage, stderr);
        return EXIT_FAILURE;
    }

    /*
     * Construct filenames
     */

    sscanf(argv[1], "%s", prefix);
    snprintf(filename1, 99, "%s.h5", prefix);
    sscanf(argv[2], "%d", &step);
    snprintf(filename2, 99, "%s.%d.h5", prefix, step);

    /*
     * Open HDF5 file (read-only). Complain if invalid.
     */

    h5file1 = H5Fopen(filename1, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5file1 < 0)
    {
        fprintf(stderr, "Could not open HDF5 file \"%s\"\n", filename1);
        return EXIT_FAILURE;
    }

    /*
     * Read model parameters from file1
     */

    input = H5Gopen(h5file1, "input", H5P_DEFAULT);
    if (input < 0)
    {
        fprintf(stderr, "Could not open /input group in \"%s\"\n", filename1);
        status = H5Fclose(h5file1);
        return EXIT_FAILURE;
    }

    status = get_attribute_int(input, "nproc_surf", &caps);
    status = get_attribute_int(input, "nprocx", &nprocx);
    status = get_attribute_int(input, "nprocy", &nprocy);
    status = get_attribute_int(input, "nprocz", &nprocz);
    status = get_attribute_int(input, "nodex", &nodex);
    status = get_attribute_int(input, "nodey", &nodey);
    status = get_attribute_int(input, "nodez", &nodez);

    status = H5Gclose(input);
    status = H5Fclose(h5file1);

    /*
     * Open HDF5 file (read-only). Complain if invalid.
     */

    h5file2 = H5Fopen(filename2, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5file2 < 0)
    {
        fprintf(stderr, "Could not open HDF5 file \"%s\"\n", filename2);
        return EXIT_FAILURE;
    }

    input = H5Gopen(h5file2, "/", H5P_DEFAULT);
    status = get_attribute_float(input, "time", &time);
    status = H5Gclose(input);

    /*
     * Read data from file1 and file2
     */

    velocity    = open_field(h5file2, "velocity");
    temperature = open_field(h5file2, "temperature");


    /* Iterate over caps */
    for(cap = 0; cap < caps; cap++)
    {
	int px, py, pz;
	int nno = nodex * nodey * nodez;

	/* Read data from HDF5 file. */
	read_field(h5file2, velocity, cap);
	read_field(h5file2, temperature, cap);

	for (py = 0; py < nprocy; py++)
	    for (px = 0; px < nprocx; px++)
		for (pz = 0; pz < nprocz; pz++)
		{
		    int rank = pz + px*nprocz + py*nprocz*nprocx
			     + cap*nprocz*nprocx*nprocy;
		    int lx = (nodex - 1) / nprocx + 1;
		    int ly = (nodey - 1) / nprocy + 1;
		    int lz = (nodez - 1) / nprocz + 1;
		    int sx = px * (lx - 1);
		    int sy = py * (ly - 1);
		    int sz = pz * (lz - 1);

		    char filename[100];
		    FILE *file;
		    int i, j, k;

		    snprintf(filename, 99, "%s.velo.%d.%d",
			     prefix, rank, step);
		    fprintf(stderr, "Writing %s\n", filename);

		    file = fopen(filename, "w");
		    fprintf(file, "%d %d %.5e\n", step, nno, time);
		    fprintf(file, "%3d %7d\n", 1, nno);

		    /* Traverse data in Citcom order */
		    for(j = sy; j < sy+ly; j++)
			for(i = sx; i < sx+lx; i++)
			    for(k = sz; k < sz+lz; k++)
			    {
				int n = k + j*nodez + i*nodez*nodey;
				fprintf(file, "%.6e %.6e %.6e %.6e\n",
					velocity->data[3*n+0],
					velocity->data[3*n+1],
					velocity->data[3*n+2],
					temperature->data[n]);
			    }

		    fclose(file);

		}
    }

    status = close_field(velocity);
    status = close_field(temperature);

    status = H5Fclose(h5file2);

    return 0;
}
