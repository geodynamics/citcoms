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

/* Routines to write the output of the finite element cycles into an
 * HDF5 file.
 */


#include <stdlib.h>
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "output_h5.h"

#ifdef USE_HDF5
#include "pytables.h"
#endif


void h5output_coord(struct All_variables *);
void h5output_velocity(struct All_variables *, int);
void h5output_temperature(struct All_variables *, int);
void h5output_viscosity(struct All_variables *, int);
void h5output_pressure(struct All_variables *, int);
void h5output_stress(struct All_variables *, int);
void h5output_material(struct All_variables *);
void h5output_tracer(struct All_variables *, int);
void h5output_surf_botm(struct All_variables *, int);
void h5output_surf_botm_pseudo_surf(struct All_variables *, int);
void h5output_ave_r(struct All_variables *, int);


extern void parallel_process_termination();
extern void heat_flux(struct All_variables *);
extern void get_STD_topo(struct All_variables *, float**, float**,
                         float**, float**, int);


/****************************************************************************
 * Functions that control which data is saved to output file(s).            *
 * These represent possible choices for (E->output) function pointer.       *
 ****************************************************************************/

void h5output(struct All_variables *E, int cycles)
{
    if (cycles == 0) {
        h5output_coord(E);
        h5output_material(E);
    }

    h5output_velocity(E, cycles);
    h5output_temperature(E, cycles);
    h5output_viscosity(E, cycles);

    if(E->control.pseudo_free_surf)
    {
        if(E->mesh.topvbc == 2)
            h5output_surf_botm_pseudo_surf(E, cycles);
    }
    else
        h5output_surf_botm(E, cycles);

    if(E->control.tracer==1)
        h5output_tracer(E, cycles);

    //h5output_stress(E, cycles);
    //h5output_pressure(E, cycles);

    /* disable horizontal average h5output   by Tan2 */
    /* h5output_ave_r(E, cycles); */

    return;
}


void h5output_pseudo_surf(struct All_variables *E, int cycles)
{

    if (cycles == 0)
    {
        h5output_coord(E);
        h5output_material(E);
    }

    h5output_velocity(E, cycles);
    h5output_temperature(E, cycles);
    h5output_viscosity(E, cycles);
    h5output_surf_botm_pseudo_surf(E, cycles);

    if(E->control.tracer==1)
        h5output_tracer(E, cycles);

    //h5output_stress(E, cycles);
    //h5output_pressure(E, cycles);

    /* disable horizontal average h5output   by Tan2 */
    /* h5output_ave_r(E, cycles); */

    return;
}


/****************************************************************************
 * Functions to initialize and finalize access to HDF5 output file.         *
 * Responsible for creating all necessary groups, attributes, and arrays.   *
 ****************************************************************************/

void h5output_open(struct All_variables *E)
{
#ifdef USE_HDF5

#endif
}

void h5output_close(struct All_variables *E)
{
#ifdef USE_HDF5

#endif
}


/****************************************************************************
 * Functions to save specific physical quantities as HDF5 arrays.           *
 ****************************************************************************/

void h5output_coord(struct All_variables *E)
{
#ifdef USE_HDF5

#endif
}

void h5output_viscosity(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_velocity(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_temperature(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_pressure(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_stress(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_material(struct All_variables *E)
{
#ifdef USE_HDF5

#endif
}

void h5output_tracer(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_surf_botm(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_surf_botm_pseudo_surf(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

void h5output_avg_r(struct All_variables *E, int cycles)
{
#ifdef USE_HDF5

#endif
}

