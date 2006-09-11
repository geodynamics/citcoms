/*
 * h5steps.c by Luis Armendariz.
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
#include <getopt.h>
#include "hdf5.h"

static int help_flag;
static int verbose_flag;
static int frames_flag;
static int steps_flag;

static herr_t read_steps(hid_t file_id, int **steps, int *numsteps);

int main(int argc, char *argv[])
{
    hid_t h5file;
    herr_t status;

    int i;
    int *steps;
    int numsteps;

    int c;
    int option_index;

    static struct option long_options[] = 
    {
        {"steps",   no_argument, 0, 's'},
        {"frames",  no_argument, 0, 'f'},
        {"verbose", no_argument, &verbose_flag, 1},
        {"help",    no_argument, &help_flag,    1},
        {0, 0, 0, 0}
    };

    help_flag = 0;
    verbose_flag = 0;
    frames_flag = 0;
    steps_flag = 1;

    /* Parse commandline options */
    for(;;)
    {
        c = getopt_long(argc, argv, "sfvh", long_options, &option_index);
        if (c == -1)
            break;
        switch (c)
        {
            case 's':
                steps_flag = 1;
                frames_flag = 0;
                break;
            case 'f':
                frames_flag = 1;
                steps_flag = 0;
                break;
            case 'h':
                help_flag = 1;
                break;
            case 'v':
                verbose_flag = 1;
                break;
        }
    }

    /* DEBUG
    printf("optind = %d\n", optind);
    printf("argc = %d\n", argc);
    for(i = 0; i < argc; i++)
        printf("argv[%d] = %s\n", i, argv[i]);
    // */

    if (help_flag || argc == 1)
    {
        fprintf(stderr, "Usage: %s file.h5\n", argv[0]);
        return EXIT_FAILURE;
    }

    /*
     * Open HDF5 file (read-only). Complain if invalid.
     */
    h5file = H5Fopen(argv[optind], H5F_ACC_RDONLY, H5P_DEFAULT);
    if (h5file < 0)
    {
        fprintf(stderr, "Could not open HDF5 file \"%s\"\n", argv[1]);
        return EXIT_FAILURE;
    }

    /*
     * Read steps from file
     */
    
    status = read_steps(h5file, &steps, &numsteps);

    if (verbose_flag)
        printf("Found %d Frames in HDF5 file \"%s\"\n", numsteps, argv[optind]);

    if (frames_flag)
    {
        if (verbose_flag)
        {
            printf("\n");
            printf("\tframes = [");
            for(i = 0; i < numsteps; i++)
                printf(" %d%c", i, (i < numsteps-1) ? ',' : ' ');
            printf("]\n\n");
        }
        else
        {
            for(i = 0; i < numsteps; i++)
                printf("%d ", i);
            printf("\n");
        }
    }

    if (steps_flag)
    {
        if (verbose_flag)
        {
            printf("\n");
            printf("\tframes = range(0,%d)\n", numsteps);
            printf("\tsteps  = [");
            for(i = 0; i < numsteps; i++)
                printf(" %d%c", steps[i], (i < numsteps-1) ? ',' : ' ');
            printf("]\n\n");
        }
        else
        {
            for(i = 0; i < numsteps; i++)
                printf("%d ", steps[i]);
            printf("\n");
        }
    }

    status = H5Fclose(h5file);
    free(steps);

    return EXIT_SUCCESS;
}

static herr_t read_steps(hid_t file_id, int **steps, int *numsteps)
{
    int rank;
    hsize_t dims;
    
    hid_t typeid;
    hid_t dataspace;
    hid_t dataset;

    herr_t status;

    dataset = H5Dopen(file_id, "time");
    
    dataspace = H5Dget_space(dataset);

    typeid = H5Tcreate(H5T_COMPOUND, sizeof(int));
    status = H5Tinsert(typeid, "step", 0, H5T_NATIVE_INT);

    rank = H5Sget_simple_extent_dims(dataspace, &dims, NULL);

    *numsteps = (int)dims;
    *steps = (int *)malloc(dims * sizeof(int));
    
    status = H5Dread(dataset, typeid, H5S_ALL, H5S_ALL, H5P_DEFAULT, *steps);
    
    status = H5Tclose(typeid);
    status = H5Sclose(dataspace);
    status = H5Dclose(dataset);

    return 0;
}
