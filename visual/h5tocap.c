/*
 * h5tocap.c by Luis Armendariz and Eh Tan.
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
    "Convert the CitcomS HDF5 output files to ASCII files,\n"
    "with the format of the combined cap files\n"
    "\n"
    "Usage: h5tocap modelname step1 [step2 [...] ]\n";


static int convert(field_t **all_coord, int ncaps, char *prefix, int step);
static void output(const char *filename, field_t *coord, field_t *velocity,
		   field_t *temperature, field_t *viscosity);


int main(int argc, char *argv[])
{
    char filename[100];
    char prefix[100];

    hid_t h5file;
    hid_t input;
    herr_t status;

    int ncaps;
    int cap;

    int *steps;
    int n, nsteps;

    field_t *all_coord[12];


    /************************************************************************
     * Parse command-line parameters.                                       *
     ************************************************************************/

    /*
     * Check number of arguments
     */

    if (argc < 3)
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
     * Read modelname
     */

    sscanf(argv[1], "%s", prefix);

    /*
     * Read step(s) from argv[2:]
     */

    /* Allocate at least one step (we know argc > 2) */
    nsteps = argc - 2;
    steps = (int *) malloc(nsteps * sizeof(int));

    /* Convert argv[2:] into int array */
    for(n = 2; n < argc; n++)
    {
	char *endptr;
	steps[n-2] = (int)strtol(argv[n], &endptr, 10);
        if (!(argv[n][0] != '\0' && *endptr == '\0'))
        {
            fprintf(stderr, "Error: Invalid step \"%s\"\n", argv[n]);
            return EXIT_FAILURE;
        }
    }

    /*
     * Open HDF5 file (read-only). Complain if invalid.
     */

    snprintf(filename, 99, "%s.h5", prefix);
    h5file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5file < 0)
    {
        fprintf(stderr, "Could not open HDF5 file \"%s\"\n", filename);
        return EXIT_FAILURE;
    }

    /*
     * Read model parameters from file
     */

    input = H5Gopen(h5file, "input", H5P_DEFAULT);
    if (input < 0)
    {
        fprintf(stderr, "Could not open /input group in \"%s\"\n", filename);
        status = H5Fclose(h5file);
        return EXIT_FAILURE;
    }

    status = get_attribute_int(input, "nproc_surf", &ncaps);
    status = H5Gclose(input);

    /*
     * Read coordinate of all caps
     */

    for(cap = 0; cap < ncaps; cap++)
    {
	all_coord[cap] = open_field(h5file, "coord");
	read_field(h5file, all_coord[cap], cap);
    }
    status = H5Fclose(h5file);

    /*
     * Convert files for each step
     */
    for(n = 0; n < nsteps; n++)
	convert(all_coord, ncaps, prefix, steps[n]);


    /* Release resources. */
    for(cap = 0; cap < ncaps; cap++)
	status = close_field(all_coord[cap]);

    free(steps);


    return EXIT_SUCCESS;
}


static int convert(field_t **all_coord, int ncaps, char *prefix, int step)
{
    char filename[100];
    hid_t h5file;
    herr_t status;

    field_t *velocity;
    field_t *temperature;
    field_t *viscosity;

    int cap;

    /*
     * Open HDF5 file (read-only). Complain if invalid.
     */

    snprintf(filename, 99, "%s.%d.h5", prefix, step);
    h5file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5file < 0)
    {
        fprintf(stderr, "Could not open HDF5 file \"%s\"\n", filename);
        return EXIT_FAILURE;
    }

    /*
     * Read data from file
     */

    velocity    = open_field(h5file, "velocity");
    temperature = open_field(h5file, "temperature");
    viscosity   = open_field(h5file, "viscosity");

    for(cap = 0; cap < ncaps; cap++)
    {
	char outfile[100];

	read_field(h5file, velocity, cap);
	read_field(h5file, temperature, cap);
	read_field(h5file, viscosity, cap);

	snprintf(outfile, 99, "%s.cap%02d.%d", prefix, cap, step);
	output(outfile, all_coord[cap], velocity, temperature, viscosity);
    }

    /* Release resources. */
    status = close_field(velocity);
    status = close_field(temperature);
    status = close_field(viscosity);
    status = H5Fclose(h5file);

    return 0;
}


static void output(const char *filename, field_t *coord, field_t *velocity,
		   field_t *temperature, field_t *viscosity)
{
    int i, j, k;
    int nodex = coord->dims[1];
    int nodey = coord->dims[2];
    int nodez = coord->dims[3];

    FILE *file = fopen(filename, "w");

    fprintf(stderr, "Writing %s\n", filename);

    fprintf(file, "%d x %d x %d\n", nodex, nodey, nodez);

    /* Traverse data in Citcom order */
    for(j = 0; j < nodey; j++)
	for(i = 0; i < nodex; i++)
	    for(k = 0; k < nodez; k++)
            {
		int n = k + j*nodez + i*nodez*nodey;
		fprintf(file, "%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
			coord->data[3*n+0],
			coord->data[3*n+1],
			coord->data[3*n+2],
			velocity->data[3*n+0],
			velocity->data[3*n+1],
			velocity->data[3*n+2],
			temperature->data[n],
			viscosity->data[n]);
	    }

    fclose(file);
}

/* vim:noet:ts=8 sw=8
 */
