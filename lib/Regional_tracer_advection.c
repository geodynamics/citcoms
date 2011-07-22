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
#include "composition_related.h"
#include "parallel_related.h"


static void write_trace_instructions(struct All_variables *E);
static void make_mesh_ijk(struct All_variables *E);
static void put_found_tracers(struct All_variables *E,
                              int recv_size, double *recv,
                              int j);
int isearch_neighbors(double *array, int nsize,
                      double a, int hint);
int isearch_all(double *array, int nsize, double a);


void regional_tracer_setup(struct All_variables *E)
{

    char output_file[255];
    void get_neighboring_caps();
    double CPU_time0();
    double begin_time = CPU_time0();

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

    /* Fixed positions in tracer array */
    /* Flavor is always in extraq position 0  */
    /* Current coordinates are always kept in basicq positions 0-5 */
    /* Other positions may be used depending on science being done */

    write_trace_instructions(E);

    /* The bounding box of neiboring processors */
    get_neighboring_caps(E);

    make_mesh_ijk(E);

    if (E->composition.on)
        composition_setup(E);

    fprintf(E->trace.fpt, "Tracer intiailization takes %f seconds.\n",
            CPU_time0() - begin_time);

    return;
}


/**** WRITE TRACE INSTRUCTIONS ***************/
static void write_trace_instructions(struct All_variables *E)
{
    int i;

    fprintf(E->trace.fpt,"\nTracing Activated! (proc: %d)\n",E->parallel.me);
    fprintf(E->trace.fpt,"   Allen K. McNamara 12-2003\n\n");

    if (E->trace.ic_method==0) {
        fprintf(E->trace.fpt,"Generating New Tracer Array\n");
        fprintf(E->trace.fpt,"Tracers per element: %d\n",E->trace.itperel);
    }
    if (E->trace.ic_method==1) {
        fprintf(E->trace.fpt,"Reading tracer file %s\n",E->trace.tracer_file);
    }
    if (E->trace.ic_method==2) {
        fprintf(E->trace.fpt,"Read individual tracer files\n");
    }

    fprintf(E->trace.fpt,"Number of tracer flavors: %d\n", E->trace.nflavors);

    if (E->trace.nflavors && E->trace.ic_method==0) {
        fprintf(E->trace.fpt,"Initialized tracer flavors by: %d\n", E->trace.ic_method_for_flavors);
        if (E->trace.ic_method_for_flavors == 0) {
            fprintf(E->trace.fpt,"Layered tracer flavors\n");
            for (i=0; i<E->trace.nflavors-1; i++)
                fprintf(E->trace.fpt,"Interface Height: %d %f\n",i,E->trace.z_interface[i]);
        }
#ifdef USE_GGRD
	else if((E->trace.ic_method_for_flavors == 1)||(E->trace.ic_method_for_flavors == 99)) {
	  /* ggrd modes 1 and 99 (99 is override for restart) */
	  fprintf(stderr,"ggrd regional flavors not implemented\n");
          fprintf(E->trace.fpt,"ggrd not implemented et for regional, flavor method= %d\n",
		  E->trace.ic_method_for_flavors);
	  fflush(E->trace.fpt);
	  parallel_process_termination();
	}
#endif
        else {
            fprintf(E->trace.fpt,"Sorry-This IC methods for Flavors are Unavailable %d\n",E->trace.ic_method_for_flavors);
            fflush(E->trace.fpt);
            parallel_process_termination();
        }
    }

    for (i=0; i<E->trace.nflavors-2; i++) {
        if (E->trace.z_interface[i] < E->trace.z_interface[i+1]) {
            fprintf(E->trace.fpt,"Sorry - The %d-th z_interface is smaller than the next one.\n", i);
            fflush(E->trace.fpt);
            parallel_process_termination();
        }
    }



    /* more obscure stuff */

    fprintf(E->trace.fpt,"Box Cushion: %f\n",E->trace.box_cushion);
    fprintf(E->trace.fpt,"Number of Basic Quantities: %d\n",
            12);
    fprintf(E->trace.fpt,"Number of Extra Quantities: %d\n",
            1);
    fprintf(E->trace.fpt,"Total Number of Tracer Quantities: %d\n",
            13);



    if (E->trace.itracer_warnings==0) {
        fprintf(E->trace.fpt,"\n WARNING EXITS ARE TURNED OFF! TURN THEM ON!\n");
        fprintf(stderr,"\n WARNING EXITS ARE TURNED OFF! TURN THEM ON!\n");
        fflush(E->trace.fpt);
    }

    write_composition_instructions(E);


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
			  CartesianCoord dummy,
			  SphericalCoord sc)
{
    int e, i, j, k;
    int ii, jj, kk;
    int elx, ely, elz;

    elx = E->lmesh.elx;
    ely = E->lmesh.ely;
    elz = E->lmesh.elz;

    //TODO: take care of south west bound


    /* Search neighboring elements if the previous element is known */
    if (iprevious_element > 0) {
	e = iprevious_element - 1;
	k = e % elz;
	i = (e / elz) % elx;
	j = e / (elz*elx);

	ii = isearch_neighbors(E->trace.x_space, elx+1, sc._theta, i);
	jj = isearch_neighbors(E->trace.y_space, ely+1, sc._phi, j);
	kk = isearch_neighbors(E->trace.z_space, elz+1, sc._rad, k);

        if (ii>=0 && jj>=0 && kk>=0)
            return jj*elx*elz + ii*elz + kk + 1;
    }

    /* Search all elements if either the previous element is unknown */
    /* or failed to find in the neighboring elements                 */
    ii = isearch_all(E->trace.x_space, elx+1, sc._theta);
    jj = isearch_all(E->trace.y_space, ely+1, sc._phi);
    kk = isearch_all(E->trace.z_space, elz+1, sc._rad);

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
                        SphericalCoord sc, double junk)
{
    double theta_min, theta_max;
    double phi_min, phi_max;

    /* corner 2 is the north-west corner */
    /* corner 4 is the south-east corner */

    theta_min = E->trace.theta_cap[icap][2];
    theta_max = E->trace.theta_cap[icap][4];

    phi_min = E->trace.phi_cap[icap][2];
    phi_max = E->trace.phi_cap[icap][4];

    if ((sc._theta >= theta_min) && (sc._theta < theta_max) &&
        (sc._phi >= phi_min) && (sc._phi < phi_max))
        return 1;

    //TODO: deal with south west bounds
    return 0;
}


void regional_get_shape_functions(struct All_variables *E,
                                  double shp[9], int nelem,
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

    shp[1] = (dx-tr_dx) * (dy-tr_dy) * (dz-tr_dz) / volume;
    shp[2] = tr_dx      * (dy-tr_dy) * (dz-tr_dz) / volume;
    shp[3] = tr_dx      * tr_dy      * (dz-tr_dz) / volume;
    shp[4] = (dx-tr_dx) * tr_dy      * (dz-tr_dz) / volume;
    shp[5] = (dx-tr_dx) * (dy-tr_dy) * tr_dz      / volume;
    shp[6] = tr_dx      * (dy-tr_dy) * tr_dz      / volume;
    shp[7] = tr_dx      * tr_dy      * tr_dz      / volume;
    shp[8] = (dx-tr_dx) * tr_dy      * tr_dz      / volume;

    /** debug **
    fprintf(E->trace.fpt, "dr=(%e,%e,%e)  tr_dr=(%e,%e,%e)\n",
            dx, dy, dz, tr_dx, tr_dy, tr_dz);
    fprintf(E->trace.fpt, "shp: %e %e %e %e %e %e %e %e\n",
            shp[1], shp[2], shp[3], shp[4], shp[5], shp[6], shp[7], shp[8]);
    fprintf(E->trace.fpt, "sum(shp): %e\n",
            shp[1]+ shp[2]+ shp[3]+ shp[4]+ shp[5]+ shp[6]+ shp[7]+ shp[8]);
    fflush(E->trace.fpt);
    /**/
    return;
}


double regional_interpolate_data(struct All_variables *E,
                                 double shp[9], double data[9])
{
    int n;
    double result = 0;

    for(n=1; n<=8; n++)
        result += data[n] * shp[n];

    return result;
}


/******** GET VELOCITY ***************************************/

CartesianCoord regional_get_velocity(struct All_variables *E,
                           int m, int nelem,
                           SphericalCoord sc)
{
    void velo_from_element_d();

    double shp[9], VV[4][9], tmp;
    int n, d, node;
    const int sphere_key = 0;
	double velocity_vector[4];

    /* get shape functions at (theta, phi, rad) */
    regional_get_shape_functions(E, shp, nelem, sc._theta, sc._phi, sc._rad);


    /* get cartesian velocity */
    velo_from_element_d(E, VV, m, nelem, sphere_key);


    /*** comment by Tan2 1/25/2005
         Interpolate the velocity on the tracer position
    ***/

    for(d=1; d<=3; d++)
        velocity_vector[d] = 0;


    for(d=1; d<=3; d++) {
        for(n=1; n<=8; n++)
            velocity_vector[d] += VV[d][n] * shp[n];
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
        tmp += E->sx[m][1][E->ien[m][nelem].node[n]] * shp[n];

    fprintf(E->trace.fpt, "THETA: %e -> %e\n", theta, tmp);

    fflush(E->trace.fpt);
    /**/

    return CartesianCoord(velocity_vector[1], velocity_vector[2], velocity_vector[3]);
}


void regional_keep_within_bounds(struct All_variables *E,
                                 CartesianCoord &cc,
                                 SphericalCoord &sc)
{
    int changed = 0;

    if (sc._theta > E->control.theta_max - E->trace.box_cushion) {
        sc._theta = E->control.theta_max - E->trace.box_cushion;
        changed = 1;
    }

    if (sc._theta < E->control.theta_min + E->trace.box_cushion) {
        sc._theta = E->control.theta_min + E->trace.box_cushion;
        changed = 1;
    }

    if (sc._phi > E->control.fi_max - E->trace.box_cushion) {
        sc._phi = E->control.fi_max - E->trace.box_cushion;
        changed = 1;
    }

    if (sc._phi < E->control.fi_min + E->trace.box_cushion) {
        sc._phi = E->control.fi_min + E->trace.box_cushion;
        changed = 1;
    }

    if (sc._rad > E->sphere.ro - E->trace.box_cushion) {
        sc._rad = E->sphere.ro - E->trace.box_cushion;
        changed = 1;
    }

    if (sc._rad < E->sphere.ri + E->trace.box_cushion) {
        sc._rad = E->sphere.ri + E->trace.box_cushion;
        changed = 1;
    }

    if (changed)
        cc = sc.toCartesian();


    return;
}


void regional_lost_souls(struct All_variables *E)
{
    /* This part only works if E->sphere.caps_per_proc==1 */
    const int j = 1;
    int lev = E->mesh.levmax;

    int i, d, kk;
    int max_send_size, isize, itemp_size;

    int ngbr_rank[6+1];

    double bounds[3][2];
    double *send[2];
    double *recv[2];

    int ipass;

    MPI_Status status[4];
    MPI_Request request[4];

	Tracer temp_tracer;
	TracerList::iterator	tr;
	
    double CPU_time0();
    double begin_time = CPU_time0();

    E->trace.istat_isend = E->trace.escaped_tracers[j].size();

    /* the bounding box */
    for (d=0; d<E->mesh.nsd; d++) {
        bounds[d][0] = E->sx[j][d+1][1];
        bounds[d][1] = E->sx[j][d+1][E->lmesh.nno];
    }

    /* set up ranks for neighboring procs */
    /* if ngbr_rank is -1, there is no neighbor on this side */
    ipass = 1;
    for (kk=1; kk<=6; kk++) {
        if (E->parallel.NUM_PASS[lev][j].bound[kk] == 1) {
            ngbr_rank[kk] = E->parallel.PROCESSOR[lev][j].pass[ipass];
            ipass++;
        }
        else {
            ngbr_rank[kk] = -1;
        }
    }

    /* debug *
    for (kk=1; kk<=E->trace.istat_isend; kk++) {
        fprintf(E->trace.fpt, "tracer#=%d xx=(%g,%g,%g)\n", kk,
                E->trace.rlater[j][0][kk],
                E->trace.rlater[j][1][kk],
                E->trace.rlater[j][2][kk]);
    }

    for (d=0; d<E->mesh.nsd; d++) {
        fprintf(E->trace.fpt, "bounds(dim=%d) = (%e, %e)\n",
                d, bounds[d][0], bounds[d][1]);
    }

    for (kk=1; kk<=6; kk++) {
        fprintf(E->trace.fpt, "pass=%d  neighbor_rank=%d\n",
                kk, ngbr_rank[kk]);
    }
    fflush(E->trace.fpt);
    parallel_process_sync(E);
    /**/


    /* Allocate Maximum Memory to Send Arrays */
    max_send_size = citmax(2*E->trace.escaped_tracers[j].size(), E->trace.tracers[j].size()/100);
    itemp_size = max_send_size * temp_tracer.size();

    if ((send[0] = (double *)malloc(itemp_size*sizeof(double)))
        == NULL) {
        fprintf(E->trace.fpt,"Error(lost souls)-no memory (u388)\n");
        fflush(E->trace.fpt);
        exit(10);
    }
    if ((send[1] = (double *)malloc(itemp_size*sizeof(double)))
        == NULL) {
        fprintf(E->trace.fpt,"Error(lost souls)-no memory (u389)\n");
        fflush(E->trace.fpt);
        exit(10);
    }


    for (d=0; d<E->mesh.nsd; d++) {
        int original_size = E->trace.escaped_tracers[j].size();
        int idb;
        int isend[2], irecv[2];
        isend[0] = isend[1] = 0;


        /* move out-of-bound tracers to send array */
        for (tr=E->trace.escaped_tracers[j].begin();tr!=E->trace.escaped_tracers[j].end();) {
            double coord;

            /* Is the tracer within the bounds in the d-th dimension */
			switch (d) {
				case 0:
					coord = tr->theta();
					break;
				case 1:
					coord = tr->phi();
					break;
				case 2:
					coord = tr->rad();
					break;
			}

            if (coord < bounds[d][0]) {
				tr->writeToMem(&send[0][isend[0] * temp_tracer.size()]);
				isend[0]++;
				tr = E->trace.escaped_tracers[j].erase(tr);
            }
            else if (coord >= bounds[d][1]) {
				tr->writeToMem(&send[1][isend[1] * temp_tracer.size()]);
				isend[1]++;
				tr = E->trace.escaped_tracers[j].erase(tr);
            }
            else {
                /* check next tracer */
                tr++;
            }

            /* reallocate send if size too small */
            if ((isend[0] > max_send_size - 5) ||
                (isend[1] > max_send_size - 5)) {

                isize = max_send_size + max_send_size/4 + 10;
                itemp_size = isize * temp_tracer.size();

                if ((send[0] = (double *)realloc(send[0],
                                                 itemp_size*sizeof(double)))
                    == NULL) {
                    fprintf(E->trace.fpt,"Error(lost souls)-no memory (s4)\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }
                if ((send[1] = (double *)realloc(send[1],
                                                 itemp_size*sizeof(double)))
                    == NULL) {
                    fprintf(E->trace.fpt,"Error(lost souls)-no memory (s5)\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }

                fprintf(E->trace.fpt,"Expanding physical memory of send to "
                        "%d from %d\n",
                        isize, max_send_size);

                max_send_size = isize;
            }


        } /* end of while kk */


        /* check the total # of tracers is conserved */
        if ((isend[0] + isend[1] + E->trace.escaped_tracers[j].size()) != original_size) {
            fprintf(E->trace.fpt, "original_size: %d, rlater_size: %d, "
                    "send_size: %d\n",
                    original_size, E->trace.escaped_tracers[j].size(), kk);
        }


        /** debug **
        for (i=0; i<2; i++) {
            for (kk=0; kk<isend[i]; kk++) {
                fprintf(E->trace.fpt, "dim:%d side:%d kk=%d coord[kk]=%e\n",
                        d, i, kk,
                        send[i][kk*E->trace.number_of_tracer_quantities+d]);
            }
        }
        fflush(E->trace.fpt);
        /**/


        /* Send info to other processors regarding number of send tracers */

        /* check whether there is a neighbor in this pass*/
        idb = 0;
        for (i=0; i<2; i++) {
            int target_rank;
            kk = d*2 + i + 1;
            target_rank = ngbr_rank[kk];
            if (target_rank >= 0) {
                MPI_Isend(&isend[i], 1, MPI_INT, target_rank,
                          11, E->parallel.world, &request[idb++]);

                MPI_Irecv(&irecv[i], 1, MPI_INT, target_rank,
                          11, E->parallel.world, &request[idb++]);
            }
            else {
                irecv[i] = 0;
            }
        } /* end of for i */


        /* Wait for non-blocking calls to complete */
        MPI_Waitall(idb, request, status);


        /** debug **
        for (i=0; i<2; i++) {
            int target_rank;
            kk = d*2 + i + 1;
            target_rank = ngbr_rank[kk];
            if (target_rank >= 0) {
                fprintf(E->trace.fpt, "%d: %d send %d to proc %d\n",
                        d, i, isend[i], target_rank);
                fprintf(E->trace.fpt, "%d: %d recv %d from proc %d\n",
                        d, i, irecv[i], target_rank);
            }
        }
        parallel_process_sync(E);
        /**/

        /* Allocate memory in receive arrays */
        for (i=0; i<2; i++) {
            isize = irecv[i] * temp_tracer.size();
            itemp_size = citmax(1, isize);

            if ((recv[i] = (double *)malloc(itemp_size*sizeof(double)))
                == NULL) {
                fprintf(E->trace.fpt, "Error(lost souls)-no memory (c721)\n");
                fflush(E->trace.fpt);
                exit(10);
            }
        }


        /* Now, send the tracers to proper procs */
        idb = 0;
        for (i=0; i<2; i++) {
            int target_rank;
            kk = d*2 + i + 1;
            target_rank = ngbr_rank[kk];
            if (target_rank >= 0) {
                isize = isend[i] * temp_tracer.size();
                MPI_Isend(send[i], isize, MPI_DOUBLE, target_rank,
                          12, E->parallel.world, &request[idb++]);

                isize = irecv[i] * temp_tracer.size();
                MPI_Irecv(recv[i], isize, MPI_DOUBLE, target_rank,
                          12, E->parallel.world, &request[idb++]);

            }
        }


        /* Wait for non-blocking calls to complete */
        MPI_Waitall(idb, request, status);


        /** debug **
        for (i=0; i<2; i++) {
            for (kk=1; kk<=irecv[i]; kk++) {
                fprintf(E->trace.fpt, "recv: %d %e %e %e\n",
                        kk,
                        recv[i][(kk-1)*E->trace.number_of_tracer_quantities],
                        recv[i][(kk-1)*E->trace.number_of_tracer_quantities+1],
                        recv[i][(kk-1)*E->trace.number_of_tracer_quantities+2]);
            }
        }
        fflush(E->trace.fpt);
        parallel_process_sync(E);
        /**/

        /* put the received tracers */
        for (i=0; i<2; i++) {
            put_found_tracers(E, irecv[i], recv[i], j);
        }


        free(recv[0]);
        free(recv[1]);

    } /* end of for d */


    /* rlater should be empty by now */
    /* ERIC: restore this later
	 if (E->trace.ilater[j] > 0) {
        fprintf(E->trace.fpt, "Error(regional_lost_souls) lost tracers\n");
        for (kk=1; kk<=E->trace.ilater[j]; kk++) {
            fprintf(E->trace.fpt, "lost #%d xx=(%e, %e, %e)\n", kk,
                    E->trace.rlater[j][0][kk],
                    E->trace.rlater[j][1][kk],
                    E->trace.rlater[j][2][kk]);
        }
        fflush(E->trace.fpt);
        exit(10);
    }*/


    /* Free Arrays */

    free(send[0]);
    free(send[1]);

    E->trace.lost_souls_time += CPU_time0() - begin_time;
    return;
}


/****************************************************************/
/* Put the received tracers in basiq & extraq, if within bounds */
/* Otherwise, append to rlater for sending to another proc      */

static void put_found_tracers(struct All_variables *E,
                              int recv_size, double *recv,
                              int j)
{
    int kk, pp;
    int ipos, ilast, inside, iel;
    double theta, phi, rad;
	Tracer new_tracer;

    for (kk=0; kk<recv_size; kk++) {
        ipos = kk * new_tracer.size();
        theta = recv[ipos];
        phi = recv[ipos + 1];
        rad = recv[ipos + 2];

        /* check whether this tracer is inside this proc */
        /* check radius first, since it is cheaper       */
        inside = icheck_processor_shell(E, j, rad);
        if (inside == 1)
            inside = regional_icheck_cap(E, 0, SphericalCoord(theta, phi, rad), rad);
        else
            inside = 0;

        /** debug **
        fprintf(E->trace.fpt, "kk=%d, inside=%d, xx=(%e, %e, %e)\n",
                kk, inside, theta, phi, rad);
        fprintf(E->trace.fpt, "before: %d %d\n",
                E->trace.ilater[j], E->trace.ntracers[j]);
        /**/

        if (inside) {

			new_tracer.readFromMem(&recv[ipos]);

            /* found the element */
            iel = regional_iget_element(E, j, -99, CartesianCoord(0, 0, 0), SphericalCoord(theta, phi, rad));

            if (iel<1) {
                fprintf(E->trace.fpt, "Error(regional lost souls) - "
                        "element not here?\n");
                fprintf(E->trace.fpt, "theta, phi, rad: %f %f %f\n",
                        theta, phi, rad);
                fflush(E->trace.fpt);
                exit(10);
            }

            new_tracer.set_ielement(iel);
			E->trace.tracers[j].push_back(new_tracer);
        }
        else {
			new_tracer.readFromMem(&recv[ipos]);
			
			E->trace.escaped_tracers[j].push_back(new_tracer);

        } /* end of if-else */

        /** debug **
        fprintf(E->trace.fpt, "after: %d %d\n",
                E->trace.ilater[j], E->trace.ntracers[j]);
        fflush(E->trace.fpt);
        /**/

    } /* end of for kk */

    return;
}
