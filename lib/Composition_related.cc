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

#include <math.h>
#include "global_defs.h"
#include "parsing.h"
#include "parallel_related.h"
#include "composition_related.h"

#include "cproto.h"


static void allocate_composition_memory(struct All_variables *E);
static void compute_elemental_composition_ratio_method(struct All_variables *E);
static void init_bulk_composition(struct All_variables *E);
static void check_initial_composition(struct All_variables *E);
static void fill_composition_from_neighbors(struct All_variables *E);


void composition_input(struct All_variables *E)
{
    int i;
    int m = E->parallel.me;

    input_boolean("chemical_buoyancy",
		  &(E->composition.ichemical_buoyancy),
		  "1,0,nomax",m);

    if (E->control.tracer && E->composition.ichemical_buoyancy) {

        /* ibuoy_type=0 (absolute method) */
        /* ibuoy_type=1 (ratio method) */

        input_int("buoy_type",&(E->composition.ibuoy_type),"1,0,nomax",m);
        if (E->composition.ibuoy_type!=1) {
            fprintf(stderr,"Terror-Sorry, only ratio method allowed now\n");
            fflush(stderr);
            parallel_process_termination();
        }

        if (E->composition.ibuoy_type==0)
            E->composition.ncomp = E->trace.nflavors;
        else if (E->composition.ibuoy_type==1)
            E->composition.ncomp = E->trace.nflavors - 1;

        E->composition.buoyancy_ratio = (double*) malloc(E->composition.ncomp
                                                         *sizeof(double));

        /* default values .... */
        for (i=0; i<E->composition.ncomp; i++)
            E->composition.buoyancy_ratio[i] = 1.0;

        input_double_vector("buoyancy_ratio", E->composition.ncomp,
                            E->composition.buoyancy_ratio,m);

    }


    /* compositional rheology */

    /* icompositional_rheology=0 (off) */
    /* icompositional_rheology=1 (on) */
    E->composition.icompositional_rheology = 0;
    /*
    input_int("compositional_rheology",
              &(E->composition.icompositional_rheology),"1,0,nomax",m);

    if (E->composition.icompositional_rheology==1) {
        input_double("compositional_prefactor",
                     &(E->composition.compositional_rheology_prefactor),
                     "1.0",m);
    }
    */

    return;
}



void composition_setup(struct All_variables *E)
{
    allocate_composition_memory(E);

    return;
}


void write_composition_instructions(struct All_variables *E)
{
    int k;

    if (E->composition.ichemical_buoyancy ||
        E->composition.icompositional_rheology)
        E->composition.on = 1;

    if (E->composition.on) {

        if (E->trace.nflavors < 1) {
            fprintf(E->trace.fpt, "Tracer flavors must be greater than 1 to track composition\n");
            parallel_process_termination();
        }

        if (!E->composition.ichemical_buoyancy)
	  fprintf(E->trace.fpt,"Passive Tracers\n");
	else
	  fprintf(E->trace.fpt,"Active Tracers\n");


        if (E->composition.ibuoy_type==1)
	  fprintf(E->trace.fpt,"Ratio Method\n");
        if (E->composition.ibuoy_type==0)
	  fprintf(E->trace.fpt,"Absolute Method\n");

        for(k=0; k<E->composition.ncomp; k++) {
            fprintf(E->trace.fpt,"Buoyancy Ratio: %f\n", E->composition.buoyancy_ratio[k]);
        }

        /*
        if (E->composition.icompositional_rheology==0) {
            fprintf(E->trace.fpt,"Compositional Rheology - OFF\n");
        }
        else if (E->composition.icompositional_rheology>0) {
            fprintf(E->trace.fpt,"Compositional Rheology - ON\n");
            fprintf(E->trace.fpt,"Compositional Prefactor: %f\n",
            E->composition.compositional_rheology_prefactor);
        }
        */

        fflush(E->trace.fpt);
    }

    return;
}


/************ FILL COMPOSITION ************************/
void fill_composition(struct All_variables *E)
{

    /* XXX: Currently, only the ratio method works here.           */
    /* Will have to come back here to include the absolute method. */

    /* ratio method */

    if (E->composition.ibuoy_type==1) {
        compute_elemental_composition_ratio_method(E);
    }

    /* absolute method */

    if (E->composition.ibuoy_type!=1) {
        fprintf(E->trace.fpt,"Error(compute...)-only ratio method now\n");
        fflush(E->trace.fpt);
        exit(10);
    }

    /* Map elemental composition to nodal points */

    map_composition_to_nodes(E);

    return;
}



static void allocate_composition_memory(struct All_variables *E)
{
    int i, j;

    for (i=0; i<E->composition.ncomp; i++) {
        E->composition.bulk_composition = (double*) malloc(E->composition.ncomp*sizeof(double));
        E->composition.initial_bulk_composition = (double*) malloc(E->composition.ncomp*sizeof(double));
        E->composition.error_fraction = (double*) malloc(E->composition.ncomp*sizeof(double));
    }


    /* for horizontal average */
    E->Have.C = (float **)malloc((E->composition.ncomp+1)*sizeof(float*));
    for (i=0; i<E->composition.ncomp; i++) {
        E->Have.C[i] = (float *)malloc((E->lmesh.noz+2)*sizeof(float));
    }


    /* allocat memory for composition fields at the nodes and elements */

    for (j=1;j<=E->sphere.caps_per_proc;j++) {
        if ((E->composition.comp_el[j]=(double **)malloc((E->composition.ncomp)*sizeof(double*)))==NULL) {
            fprintf(E->trace.fpt,"AKM(allocate_composition_memory)-no memory 8987y\n");
            fflush(E->trace.fpt);
            exit(10);
        }
        if ((E->composition.comp_node[j]=(double **)malloc((E->composition.ncomp)*sizeof(double*)))==NULL) {
            fprintf(E->trace.fpt,"AKM(allocate_composition_memory)-no memory 8988y\n");
            fflush(E->trace.fpt);
            exit(10);
        }

        for (i=0; i<E->composition.ncomp; i++) {
            if ((E->composition.comp_el[j][i]=(double *)malloc((E->lmesh.nel+1)*sizeof(double)))==NULL) {
                fprintf(E->trace.fpt,"AKM(allocate_composition_memory)-no memory 8989y\n");
                fflush(E->trace.fpt);
                exit(10);
            }

            if ((E->composition.comp_node[j][i]=(double *)malloc((E->lmesh.nno+1)*sizeof(double)))==NULL) {
                fprintf(E->trace.fpt,"AKM(allocate_composition_memory)-no memory 983rk\n");
                fflush(E->trace.fpt);
                exit(10);
            }
        }

    }

    return;
}


void init_composition(struct All_variables *E)
{
    /* XXX: Currently, only the ratio method works here.           */
    /* Will have to come back here to include the absolute method. */

    /* ratio method */
    if (E->composition.ibuoy_type==1) {
        compute_elemental_composition_ratio_method(E);
    }

    /* absolute method */
    if (E->composition.ibuoy_type!=1) {
        fprintf(E->trace.fpt,"Error(compute...)-only ratio method now\n");
        fflush(E->trace.fpt);
        exit(10);
    }

    /* for empty elements */
    check_initial_composition(E);

    /* Map elemental composition to nodal points */
    map_composition_to_nodes(E);

    init_bulk_composition(E);

    return;
}


static void check_initial_composition(struct All_variables *E)
{
    /* check empty element if using ratio method */
    if (E->composition.ibuoy_type == 1) {
        if (E->trace.istat_iempty) {
            /* using the composition of neighboring elements to determine
               the initial composition of empty elements. */
            fill_composition_from_neighbors(E);
        }
    }

    return;
}



/*********** COMPUTE ELEMENTAL COMPOSITION RATIO METHOD ***/
/*                                                        */
/* This function computes the composition per element.    */
/* The concentration of material i in an element is       */
/* defined as:                                            */
/*   (# of tracers of flavor i) / (# of all tracers)      */

static void compute_elemental_composition_ratio_method(struct All_variables *E)
{
    int i, j, e, flavor, numtracers;
    int iempty = 0;


    for (j=1; j<=E->sphere.caps_per_proc; j++) {
        for (e=1; e<=E->lmesh.nel; e++) {
            numtracers = 0;
            for (flavor=0; flavor<E->trace.nflavors; flavor++)
                numtracers += E->trace.ntracer_flavor[j][flavor][e];

            /* Check for empty entries and compute ratio.  */
            /* If no tracers are in an element, skip this element, */
            /* use previous composition. */
            if (numtracers == 0) {
                iempty++;
                /* fprintf(E->trace.fpt, "No tracer in element %d!\n", e); */
                continue;
            }

            for(i=0;i<E->composition.ncomp;i++) {
                flavor = i + 1;
                E->composition.comp_el[j][i][e] =
                    E->trace.ntracer_flavor[j][flavor][e] / (double)numtracers;
            }
        }


        if (iempty) {

            if ((1.0*iempty/E->lmesh.nel)>0.80) {
                fprintf(E->trace.fpt,"WARNING(compute_elemental...)-number of tracers is REALLY LOW\n");
                fflush(E->trace.fpt);
                if (E->trace.itracer_warnings) exit(10);
            }
        }

    } /* end j */

    E->trace.istat_iempty += iempty;

    return;
}

/********** MAP COMPOSITION TO NODES ****************/
/*                                                  */


void map_composition_to_nodes(struct All_variables *E)
{
    double *tmp[NCS];
    int i, n, kk;
    int nelem, nodenum;
    int j;


    for (j=1;j<=E->sphere.caps_per_proc;j++) {

        /* first, initialize node array */
        for(i=0;i<E->composition.ncomp;i++) {
            for (kk=1;kk<=E->lmesh.nno;kk++)
                E->composition.comp_node[j][i][kk]=0.0;
        }

        /* Loop through all elements */
        for (nelem=1;nelem<=E->lmesh.nel;nelem++) {

            /* for each element, loop through element nodes */

            /* weight composition */

            for (nodenum=1;nodenum<=8;nodenum++) {
                n = E->ien[j][nelem].node[nodenum];
                for(i=0;i<E->composition.ncomp;i++) {

                    E->composition.comp_node[j][i][n] +=
                        E->composition.comp_el[j][i][nelem]*
                        E->TWW[E->mesh.levmax][j][nelem].node[nodenum];
                }
            }

        } /* end nelem */
    } /* end j */

    for(i=0;i<E->composition.ncomp;i++) {
        for (j=1;j<=E->sphere.caps_per_proc;j++)
            tmp[j] = E->composition.comp_node[j][i];

        (E->exchange_node_d)(E,tmp,E->mesh.levmax);
    }

    /* Divide by nodal volume */
    for (j=1;j<=E->sphere.caps_per_proc;j++) {
        for(i=0;i<E->composition.ncomp;i++)
            for (kk=1;kk<=E->lmesh.nno;kk++)
                E->composition.comp_node[j][i][kk] *= E->MASS[E->mesh.levmax][j][kk];

        /* testing */
        /**
        for(i=0;i<E->composition.ncomp;i++)
            for (kk=1;kk<=E->lmesh.nel;kk++) {
                fprintf(E->trace.fpt,"%d %f\n",kk,E->composition.comp_el[j][i][kk]);
            }

        for(i=0;i<E->composition.ncomp;i++)
            for (kk=1;kk<=E->lmesh.nno;kk++) {
                fprintf(E->trace.fpt,"%d %f %f\n",kk,E->sx[j][3][kk],E->composition.comp_node[j][i][kk]);
            }
        fflush(E->trace.fpt);
        /**/

    } /* end j */

    return;
}


/****************************************************************/

static void fill_composition_from_neighbors(struct All_variables *E)
{
    int i, j, k, e, ee, n, flavor, numtracers, count;
    double *sum;
    const int n_nghbrs = 4;
    int nghbrs[n_nghbrs];
    int *is_empty;

    fprintf(E->trace.fpt,"WARNING(check_initial_composition)-number of tracers is low, %d elements contain no tracer initially\n", E->trace.istat_iempty);

    fprintf(E->trace.fpt,"Using neighboring elements for initial composition...\n");

    /* index shift for neighboring elements in horizontal direction */
    nghbrs[0] = E->lmesh.elz;
    nghbrs[1] = -E->lmesh.elz;
    nghbrs[2] = E->lmesh.elz * E->lmesh.elx;
    nghbrs[3] = -E->lmesh.elz * E->lmesh.elx;

    is_empty = (int *)calloc(E->lmesh.nel+1, sizeof(int));
    sum = (double *)malloc(E->composition.ncomp * sizeof(double));

    for (j=1; j<=E->sphere.caps_per_proc; j++) {
        /* which element is empty? */
        for (e=1; e<=E->lmesh.nel; e++) {
            numtracers = 0;
            for (flavor=0; flavor<E->trace.nflavors; flavor++)
                numtracers += E->trace.ntracer_flavor[j][flavor][e];

            if (numtracers == 0)
                is_empty[e] = 1;
        }

        /* using the average comp_el from neighboring elements */
        for (e=1; e<=E->lmesh.nel; e++) {
            if(is_empty[e]) {
                count = 0;
                for (i=0; i<E->composition.ncomp; i++)
                    sum[i] = 0.0;

                for(n=0; n<n_nghbrs; n++) {
                    ee = e + nghbrs[n];
                    /* is ee a valid element number and the elemnt is not empty? */
                    if((ee>0) && (ee<=E->lmesh.nel) && (!is_empty[ee])) {
                        count++;
                        for (i=0; i<E->composition.ncomp; i++)
                            sum[i] += E->composition.comp_el[j][i][ee];
                    }
                }

                if(count == 0) {
                    fprintf(E->trace.fpt,"Error(fill_composition_from_neighbors)-all neighboring elements are empty\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }

                for (i=0; i<E->composition.ncomp; i++)
                    E->composition.comp_el[j][i][e] = sum[i] / count;
            }
        }
    }

    free(is_empty);
    free(sum);

    fprintf(E->trace.fpt,"Done.\n");
    fflush(E->trace.fpt);
    return;
}


/*********** GET BULK COMPOSITION *******************************/

static void init_bulk_composition(struct All_variables *E)
{

    double volume;
    double *tmp[NCS];
    int i, m;
    const int ival=0;


    for (i=0; i<E->composition.ncomp; i++) {

        for (m=1;m<=E->sphere.caps_per_proc;m++)
            tmp[m] = E->composition.comp_node[m][i];

        /* ival=0 returns integral not average */
        volume = return_bulk_value_d(E,tmp,ival);

        E->composition.bulk_composition[i] = volume;
        E->composition.initial_bulk_composition[i] = volume;
    }

    return;
}


void get_bulk_composition(struct All_variables *E)
{

    double volume;
    double *tmp[NCS];
    int i, m;
    const int ival = 0;

    for (i=0; i<E->composition.ncomp; i++) {

        for (m=1;m<=E->sphere.caps_per_proc;m++)
            tmp[m] = E->composition.comp_node[m][i];

        /* ival=0 returns integral not average */
        volume = return_bulk_value_d(E,tmp,ival);

        E->composition.bulk_composition[i] = volume;

        E->composition.error_fraction[i] = (volume - E->composition.initial_bulk_composition[i]) / E->composition.initial_bulk_composition[i];
    }

    return;
}
