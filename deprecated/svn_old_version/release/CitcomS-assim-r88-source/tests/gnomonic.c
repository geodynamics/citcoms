/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */


/* Compile by: mpicc gnomonic.c -lCitcomS -L../lib -lz */

#include <math.h>
#include <stdio.h>

void spherical_to_uv2(double center[2], int len,
                      double *theta, double *phi,
                      double *u, double *v);
void uv_to_spherical(double center[2], int len,
                     double *u, double *v,
                     double *theta, double *phi);

/* test for gnomonic projection and its inverse */


int main(int argc, char **argv)
{
    #define len   6
    int i;

    double u[len], v[len];
    double center[2] = {M_PI / 2, 0};
    double theta[len] = {0.1, 0.2, 0.3, 0.3, center[0], M_PI/4};
    double phi[len] = {0.1, 0.1, 0.3, 0.4, center[1], M_PI/4};

    spherical_to_uv2(center, len, theta, phi, u, v);

    for(i=0; i<len; i++) {
        fprintf(stderr, "(%.15e, %.15e) -> (%.15e, %.15e)\n",
                theta[i], phi[i], u[i], v[i]);
    }
    fprintf(stderr, "\n\n");

    uv_to_spherical(center, len, u, v, theta, phi);

    for(i=0; i<len; i++) {
        fprintf(stderr, "(%.15e, %.15e) <- (%.15e, %.15e)\n",
                theta[i], phi[i], u[i], v[i]);
    }

    return 0;
}
