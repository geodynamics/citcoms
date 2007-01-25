/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>

#include "global_defs.h"
#include "material_properties.h"


void mat_prop_allocate(struct All_variables *E)
{
    int noz = E->lmesh.noz;
    int nno = E->lmesh.nno;
    int nel = E->lmesh.nel;

    /* reference profile of density */
    E->rho_ref = (double *) malloc((noz+1)*sizeof(double));

    /* reference profile of coefficient of thermal expansion */
    E->thermexp_ref = (double *) malloc((noz+1)*sizeof(double));

    /* reference profile of temperature */
    E->T_ref = (double *) malloc((noz+1)*sizeof(double));

    /* reference profile of d(ln(rho_ref))/dr */
    E->dlnrhodr = (double *) malloc((nel+1)*sizeof(double));
}


void reference_state(struct All_variables *E)
{
    int noz = E->lmesh.noz;
    int nel = E->lmesh.nel;
    int i;
    double r, z, tmp, T0;

    tmp = E->control.disptn_number * E->control.inv_gruneisen;
    T0 = E->data.surf_temp / E->data.ref_temperature;

    for(i=1; i<=noz; i++) {
	r = E->sx[1][3][i];
	z = 1 - r;
	E->rho_ref[i] = exp(tmp*z);
	E->thermexp_ref[i] = 1;
	E->T_ref[i] = T0 * (exp(E->control.disptn_number * z) - 1);
    }

    for(i=1; i<=nel; i++) {
        // TODO: dln(rho)/dr
        E->dlnrhodr[i] = - tmp;
    }

    if(E->parallel.me < E->parallel.nprocz)
        for(i=1; i<=noz; i++) {
            fprintf(stderr, "%d %f %f %f %f\n",
                    i+E->lmesh.nzs-1, E->sx[1][3][i], 1-E->sx[1][3][i],
                    E->rho_ref[i], E->thermexp_ref[i]);
        }
}


void density(struct All_variables *E, double *rho)
{
    int i;
    for(i=1; i<=E->lmesh.nno; i++) {
	rho[i] = 1;
    }

}
