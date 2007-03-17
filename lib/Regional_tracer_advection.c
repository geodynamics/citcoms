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



#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <mpi.h>
#include <math.h>
#include <sys/types.h>
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif

#include "element_definitions.h"
#include "global_defs.h"

static void make_mesh_ijk(struct All_variables *E);


void regional_tracer_setup(struct All_variables *E)
{
    char output_file[255];

    /* Some error control */

    if (E->sphere.caps_per_proc>1) {
            fprintf(stderr,"This code does not work for multiple caps per processor!\n");
            parallel_process_termination();
    }


    /* open tracing output file */

    sprintf(output_file,"%s.tracer_log.%d",E->control.data_file,E->parallel.me);
    E->trace.fpt=fopen(output_file,"w");


    /* reset statistical counters */

    E->trace.istat_isend=0;
    E->trace.istat_iempty=0;
    E->trace.istat_elements_checked=0;
    E->trace.istat1=0;


    /* some obscure initial parameters */
    /* This parameter specifies how close a tracer can get to the boundary */
    E->trace.box_cushion=0.00001;

    /* AKMA turn this back on after debugging */
    E->trace.itracer_warnings=1;

    /* Determine number of tracer quantities */

    /* advection_quantites - those needed for advection */
    E->trace.number_of_basic_quantities=12;

    /* extra_quantities - used for flavors, composition, etc.    */
    /* (can be increased for additional science i.e. tracing chemistry */

    E->trace.number_of_extra_quantities = 0;
    if (E->trace.nflavors > 0)
        E->trace.number_of_extra_quantities += 1;


    E->trace.number_of_tracer_quantities =
        E->trace.number_of_basic_quantities +
        E->trace.number_of_extra_quantities;


    /* Fixed positions in tracer array */
    /* Flavor is always in extraq position 0  */
    /* Current coordinates are always kept in basicq positions 0-5 */
    /* Other positions may be used depending on advection scheme and/or science being done */


    /* Some error control regarding size of pointer arrays */

    if (E->trace.number_of_basic_quantities>99) {
        fprintf(E->trace.fpt,"ERROR(initialize_trace)-increase 2nd position size of basic in tracer_defs.h\n");
        fflush(E->trace.fpt);
        parallel_process_termination();
    }
    if (E->trace.number_of_extra_quantities>99) {
        fprintf(E->trace.fpt,"ERROR(initialize_trace)-increase 2nd position size of extraq in tracer_defs.h\n");
        fflush(E->trace.fpt);
        parallel_process_termination();
    }
    if (E->trace.number_of_tracer_quantities>99) {
        fprintf(E->trace.fpt,"ERROR(initialize_trace)-increase 2nd position size of rlater in tracer_defs.h\n");
        fflush(E->trace.fpt);
        parallel_process_termination();
    }

    /* The bounding box of neiboring processors */
    get_neighboring_caps(E);


    if (E->trace.ic_method==0) {
        fprintf(E->trace.fpt,"Not ready for this inputs yet, tracer_ic_method=%d\n", E->trace.ic_method);
        fflush(E->trace.fpt);
        parallel_process_termination();
    }
    else if (E->trace.ic_method==1)
        read_tracer_file(E);
    else if (E->trace.ic_method==2)
        restart_tracers(E);
    else {
        fprintf(E->trace.fpt,"Not ready for other inputs yet\n");
        fflush(E->trace.fpt);
        parallel_process_termination();
    }

    /* total number of tracers  */

    E->trace.ilast_tracer_count = isum_tracers(E);
    fprintf(E->trace.fpt, "Sum of Tracers: %d\n", E->trace.ilast_tracer_count);


    make_mesh_ijk(E);


    /* find elements */

    find_tracers(E);


    /* count # of tracers of each flavor */

    if (E->trace.nflavors > 0)
        count_tracers_of_flavors(E);


    composition_setup(E);
    tracer_post_processing(E);

    return;
}


static void make_mesh_ijk(struct All_variables *E)
{
    int m,i,j,k,node;
    int nox,noy,noz;

    nox=E->lmesh.nox;
    noy=E->lmesh.noy;
    noz=E->lmesh.noz;

    E->trace.x_space=(double*) malloc(nox*sizeof(double));
    E->trace.y_space=(double*) malloc(noy*sizeof(double));
    E->trace.z_space=(double*) malloc(noz*sizeof(double));

    /***comment by Vlad 1/26/2005
        reading the local mesh coordinate
    ***/

    for(m=1;m<=E->sphere.caps_per_proc;m++)  {
        for(i=0;i<nox;i++)
	    E->trace.x_space[i]=E->sx[m][1][i*noz+1];

        for(j=0;j<noy;j++)
	    E->trace.y_space[j]=E->sx[m][2][j*nox*noz+1];

        for(k=0;k<noz;k++)
	    E->trace.z_space[k]=E->sx[m][3][k+1];

    }   /* end of m  */


    /* debug *
    for(i=0;i<nox;i++)
	fprintf(E->trace.fpt, "i=%d x=%e\n", i, E->trace.x_space[i]);
    for(j=0;j<noy;j++)
	fprintf(E->trace.fpt, "j=%d y=%e\n", j, E->trace.y_space[j]);
    for(k=0;k<noz;k++)
	fprintf(E->trace.fpt, "k=%d z=%e\n", k, E->trace.z_space[k]);

    /**
    fprintf(stderr, "%d\n", isearch_neighbors(E->trace.z_space, noz, 0.7, 0));
    fprintf(stderr, "%d\n", isearch_neighbors(E->trace.z_space, noz, 0.7, 1));
    fprintf(stderr, "%d\n", isearch_neighbors(E->trace.z_space, noz, 0.7, 2));
    fprintf(stderr, "%d\n", isearch_neighbors(E->trace.z_space, noz, 0.7, 3));
    fprintf(stderr, "%d\n", isearch_neighbors(E->trace.z_space, noz, 0.7, 4));

    fprintf(stderr, "%d\n", isearch_neighbors(E->trace.z_space, noz, 0.56, 0));
    fprintf(stderr, "%d\n", isearch_neighbors(E->trace.z_space, noz, 0.56, 1));
    fprintf(stderr, "%d\n", isearch_neighbors(E->trace.z_space, noz, 0.56, 2));

    fprintf(stderr, "%d\n", isearch_neighbors(E->trace.z_space, noz, 0.99, 2));
    fprintf(stderr, "%d\n", isearch_neighbors(E->trace.z_space, noz, 0.99, 3));
    fprintf(stderr, "%d\n", isearch_neighbors(E->trace.z_space, noz, 0.99, 4));

    fprintf(stderr, "%d\n", isearch_all(E->trace.z_space, noz, 0.5));
    fprintf(stderr, "%d\n", isearch_all(E->trace.z_space, noz, 1.1));
    fprintf(stderr, "%d\n", isearch_all(E->trace.z_space, noz, 0.55));
    fprintf(stderr, "%d\n", isearch_all(E->trace.z_space, noz, 1.0));
    fprintf(stderr, "%d\n", isearch_all(E->trace.z_space, noz, 0.551));
    fprintf(stderr, "%d\n", isearch_all(E->trace.z_space, noz, 0.99));
    fprintf(stderr, "%d\n", isearch_all(E->trace.z_space, noz, 0.7));
    fprintf(stderr, "%d\n", isearch_all(E->trace.z_space, noz, 0.75));
    fprintf(stderr, "%d\n", isearch_all(E->trace.z_space, noz, 0.775));
    fprintf(stderr, "%d\n", isearch_all(E->trace.z_space, noz, 0.7750001));
    parallel_process_termination();
    /**/

    return;
}


/********** IGET ELEMENT *****************************************/
/*                                                               */
/* This function returns the the real element for a given point. */
/* Returns -99 in not in this cap.                               */
/* iprevious_element, if known, is the last known element. If    */
/* it is not known, input a negative number.                     */

int regional_iget_element(struct All_variables *E,
			  int m, int iprevious_element,
			  double x, double y, double z,
			  double theta, double phi, double rad)
{
    int e, i, j, k;
    int ii, jj, kk;
    int elx, ely, elz;

    elx = E->lmesh.elx;
    ely = E->lmesh.ely;
    elz = E->lmesh.elz;

    //TODO: take care of upper bound


    /* Search neighboring elements if the previous element is known */
    if (iprevious_element > 0) {
	e = iprevious_element - 1;
	k = e % elz;
	i = (e / elz) % elx;
	j = e / (elz*elx);

	ii = isearch_neighbors(E->trace.x_space, elx+1, theta, i);
	jj = isearch_neighbors(E->trace.y_space, ely+1, phi, j);
	kk = isearch_neighbors(E->trace.z_space, elz+1, rad, k);

        if (ii>=0 && jj>=0 && kk>=0)
            return jj*elx*elz + ii*elz + kk + 1;
    }

    /* Search all elements if either the previous element is unknown */
    /* or failed to find in the neighboring elements                 */
    ii = isearch_all(E->trace.x_space, elx+1, theta);
    jj = isearch_all(E->trace.y_space, ely+1, phi);
    kk = isearch_all(E->trace.z_space, elz+1, rad);

    if (ii<0 || jj<0 || kk<0)
        return -99;

    return jj*elx*elz + ii*elz + kk + 1;
}


/* array is an ordered array of length nsize               */
/* return an index i, such that array[i] <= a < array[i+1] */
/* return -1 if not found.                                 */
/* Note that -1 is returned if a == array[nsize-1]         */
int isearch_all(double *array, int nsize, double a)
{
    int high, i, low;

    /* check the min/max bound */
    if ((a < array[0]) || (a >= array[nsize-1]))
        return -1;

    /* binary search */
    for (low=0, high=nsize-1; high-low>1;) {
        i = (high+low) / 2;
        if ( a < array[i] ) high = i;
        else low = i;
    }

    return low;
}


/* Similar the isearch_all(), but with a hint */
int isearch_neighbors(double *array, int nsize,
                      double a, int hint)
{
    /* search the nearest neighbors only */
    const int number_of_neighbors = 3;
    int neighbors[5];
    int n, i;

    neighbors[0] = hint;
    neighbors[1] = hint-1;
    neighbors[2] = hint+1;
    neighbors[3] = hint-2;
    neighbors[4] = hint+2;


    /**/
    for (n=0; n<number_of_neighbors; n++) {
        i = neighbors[n];
        if ((i >= 0) && (i < nsize-1) &&
            (a >= array[i]) && (a < array[i+1]))
            return i;
    }

    return -1;
}


/*                                                          */
/* This function serves to determine if a point lies within */
/* a given cap                                              */
/*                                                          */
int regional_icheck_cap(struct All_variables *E, int icap,
                        double theta, double phi, double rad, double junk)
{
    double theta_min, theta_max;
    double phi_min, phi_max;

    /* corner 2 is the lower-left corner */
    /* corner 4 is the upper-right corner */

    theta_min = E->trace.theta_cap[icap][2];
    theta_max = E->trace.theta_cap[icap][4];

    phi_min = E->trace.phi_cap[icap][2];
    phi_max = E->trace.phi_cap[icap][4];

    if ((theta >= theta_min) && (theta < theta_max) &&
        (phi >= phi_min) && (phi < phi_max))
        return 1;

    //TODO: deal with upper right bounds
    return 0;
}


static void get_shape_functions(struct All_variables *E,
                                double w[9], int nelem,
                                double theta, double phi, double rad)
{
    int e, i, j, k;
    int elx, ely, elz;
    double tr_dx, tr_dy, tr_dz;
    double dx, dy, dz;
    double volume;

    elx = E->lmesh.elx;
    ely = E->lmesh.ely;
    elz = E->lmesh.elz;

    e = nelem - 1;
    k = e % elz;
    i = (e / elz) % elx;
    j = e / (elz*elx);


   /*** comment by Tan2 1/25/2005
         Find the element that contains the tracer.

       node(i)     tracer              node(i+1)
         |           *                    |
         <----------->
         tr_dx

         <-------------------------------->
         dx
    ***/

    tr_dx = theta - E->trace.x_space[i];
    dx = E->trace.x_space[i+1] - E->trace.x_space[i];

    tr_dy = phi - E->trace.y_space[j];
    dy = E->trace.y_space[j+1] - E->trace.y_space[j];

    tr_dz = rad - E->trace.z_space[k];
    dz = E->trace.z_space[k+1] - E->trace.z_space[k];



    /*** comment by Tan2 1/25/2005
         Calculate shape functions from tr_dx, tr_dy, tr_dz
         This assumes linear element
    ***/


    /* compute volumetic weighting functions */
    volume = dx*dz*dy;

    w[1] = (dx-tr_dx) * (dy-tr_dy) * (dz-tr_dz) / volume;
    w[2] = tr_dx      * (dy-tr_dy) * (dz-tr_dz) / volume;
    w[3] = tr_dx      * tr_dy      * (dz-tr_dz) / volume;
    w[4] = (dx-tr_dx) * tr_dy      * (dz-tr_dz) / volume;
    w[5] = (dx-tr_dx) * (dy-tr_dy) * tr_dz      / volume;
    w[6] = tr_dx      * (dy-tr_dy) * tr_dz      / volume;
    w[7] = tr_dx      * tr_dy      * tr_dz      / volume;
    w[8] = (dx-tr_dx) * tr_dy      * tr_dz      / volume;

    /** debug **
    fprintf(E->trace.fpt, "dr=(%e,%e,%e)  tr_dr=(%e,%e,%e)\n",
            dx, dy, dz, tr_dx, tr_dy, tr_dz);
    fprintf(E->trace.fpt, "shp: %e %e %e %e %e %e %e %e\n",
            w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8]);
    fprintf(E->trace.fpt, "sum(shp): %e\n",
            w[1]+ w[2]+ w[3]+ w[4]+ w[5]+ w[6]+ w[7]+ w[8]);
    fflush(E->trace.fpt);
    /**/
    return;
}



/******** GET VELOCITY ***************************************/

void regional_get_velocity(struct All_variables *E,
                           int m, int nelem,
                           double theta, double phi, double rad,
                           double *velocity_vector)
{
    void velo_from_element_d();

    double weight[9], VV[4][9], tmp;
    int n, d, node;
    const int sphere_key = 0;

    /* get shape functions at (theta, phi, rad) */
    get_shape_functions(E, weight, nelem, theta, phi, rad);


    /* get cartesian velocity */
    velo_from_element_d(E, VV, m, nelem, sphere_key);


    /*** comment by Tan2 1/25/2005
         Interpolate the velocity on the tracer position
    ***/

    for(d=1; d<=3; d++)
        velocity_vector[d] = 0;


    for(d=1; d<=3; d++) {
        for(n=1; n<=8; n++)
            velocity_vector[d] += VV[d][n] * weight[n];
    }


    /** debug **
    for(d=1; d<=3; d++) {
        fprintf(E->trace.fpt, "VV: %e %e %e %e %e %e %e %e: %e\n",
                VV[d][1], VV[d][2], VV[d][3], VV[d][4],
                VV[d][5], VV[d][6], VV[d][7], VV[d][8],
                velocity_vector[d]);
    }

    tmp = 0;
    for(n=1; n<=8; n++)
        tmp += E->sx[m][1][E->ien[m][nelem].node[n]] * weight[n];

    fprintf(E->trace.fpt, "THETA: %e -> %e\n", theta, tmp);

    fflush(E->trace.fpt);
    /**/

    return;
}


void regional_put_lost_tracers(struct All_variables *E,
                               int isend[13][13], double *send[13][13])
{
    int j, kk, pp;
    int numtracers, ithatcap, icheck;
    int isend_position, ipos;
    int lev = E->mesh.levmax;
    double theta, phi, rad;


    for (j=1; j<=E->sphere.caps_per_proc; j++) {

        /* transfer tracers from rlater to send */

        numtracers = E->trace.ilater[j];

        for (kk=1; kk<=numtracers; kk++) {
            theta = E->trace.rlater[j][0][kk];
            phi = E->trace.rlater[j][1][kk];
            rad = E->trace.rlater[j][2][kk];


            /* first check same cap if nprocz>1 */

            if (E->parallel.nprocz>1) {
                ithatcap = 0;
                icheck = regional_icheck_cap(E, ithatcap, theta, phi, rad, rad);
                if (icheck == 1) goto foundit;

            }

            /* check neighboring caps */

            for (pp=1; pp<=E->parallel.TNUM_PASS[lev][j]; pp++) {
                ithatcap = pp;
                icheck = regional_icheck_cap(E, ithatcap, theta, phi, rad, rad);
                if (icheck == 1) goto foundit;
            }


            /* should not be here */
            if (icheck!=1) {
                fprintf(E->trace.fpt,"Error(lost souls)-should not be here\n");
                fprintf(E->trace.fpt,"theta: %f phi: %f rad: %f\n",
                        theta,phi,rad);
                icheck = regional_icheck_cap(E, 0, theta, phi, rad,rad);
                if (icheck == 1) fprintf(E->trace.fpt," icheck here!\n");
                else fprintf(E->trace.fpt,"icheck not here!\n");
                fflush(E->trace.fpt);
                exit(10);
            }

        foundit:

            isend[j][ithatcap]++;

            /* assign tracer to send */

            isend_position=(isend[j][ithatcap]-1)*E->trace.number_of_tracer_quantities;

            for (pp=0;pp<=(E->trace.number_of_tracer_quantities-1);pp++) {
                ipos=isend_position+pp;
                send[j][ithatcap][ipos]=E->trace.rlater[j][pp][kk];
            }

        } /* end kk, assigning tracers */

    } /* end j */


    return;
}
