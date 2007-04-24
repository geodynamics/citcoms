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


static void allocate_composition_memory(struct All_variables *E);
static void compute_elemental_composition_ratio_method(struct All_variables *E);
static void init_bulk_composition(struct All_variables *E);
static void check_initial_composition(struct All_variables *E);
static void map_composition_to_nodes(struct All_variables *E);


void composition_input(struct All_variables *E)
{
    int m = E->parallel.me;

    input_int("chemical_buoyancy",&(E->composition.ichemical_buoyancy),
              "1,0,nomax",m);

    if (E->composition.ichemical_buoyancy==1) {

        input_double("buoyancy_ratio",
                     &(E->composition.buoyancy_ratio),"1.0",m);

        /* ibuoy_type=0 (absolute method) */
        /* ibuoy_type=1 (ratio method) */

        input_int("buoy_type",&(E->composition.ibuoy_type),"1,0,nomax",m);
        if (E->composition.ibuoy_type!=1) {
            fprintf(stderr,"Terror-Sorry, only ratio method allowed now\n");
            fflush(stderr);
            parallel_process_termination();
        }

        input_int("reset_initial_composition",
                  &(E->composition.ireset_initial_composition),"0",m);

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

    E->composition.on = 0;
    if (E->composition.ichemical_buoyancy==1 ||
        E->composition.icompositional_rheology)
        E->composition.on = 1;

    if (E->composition.on) {

        if (E->trace.nflavors < 1) {
            fprintf(E->trace.fpt, "Tracer flavors must be greater than 1 to track composition\n");
            parallel_process_termination();
        }

        if (E->composition.ichemical_buoyancy==0)
            fprintf(E->trace.fpt,"Passive Tracers\n");

        if (E->composition.ichemical_buoyancy==1)
            fprintf(E->trace.fpt,"Active Tracers\n");


        if (E->composition.ibuoy_type==1) fprintf(E->trace.fpt,"Ratio Method\n");
        if (E->composition.ibuoy_type==0) fprintf(E->trace.fpt,"Absolute Method\n");

        fprintf(E->trace.fpt,"Buoyancy Ratio: %f\n", E->composition.buoyancy_ratio);

        if (E->composition.ireset_initial_composition==0)
            fprintf(E->trace.fpt,"Using old initial composition from tracer files\n");
        else
            fprintf(E->trace.fpt,"Resetting initial composition\n");


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
    int j;

    /* allocat memory for composition fields at the nodes and elements */

    for (j=1;j<=E->sphere.caps_per_proc;j++) {
        if ((E->composition.comp_el[j]=(double *)malloc((E->lmesh.nel+1)*sizeof(double)))==NULL) {
            fprintf(E->trace.fpt,"AKM(allocate_composition_memory)-no memory 8989y\n");
            fflush(E->trace.fpt);
            exit(10);
        }

        if ((E->composition.comp_node[j]=(double *)malloc((E->lmesh.nno+1)*sizeof(double)))==NULL) {
            fprintf(E->trace.fpt,"AKM(allocate_composition_memory)-no memory 983rk\n");
            fflush(E->trace.fpt);
            exit(10);
        }
    }

    return;
}


void init_composition(struct All_variables *E)
{
    if (E->composition.ichemical_buoyancy==1 && E->composition.ibuoy_type==1) {
        fill_composition(E);
        check_initial_composition(E);
        init_bulk_composition(E);
    }
    return;
}


static void check_initial_composition(struct All_variables *E)
{
    /* check empty element if using ratio method */
    if (E->composition.ibuoy_type == 1) {
        if (E->trace.istat_iempty) {
            fprintf(E->trace.fpt,"WARNING(check_initial_composition)-number of tracers is REALLY LOW\n");
            fflush(E->trace.fpt);
            fprintf(stderr,"WARNING(check_initial_composition)-number of tracers is REALLY LOW\n");
            exit(10);
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
    int j, e, flavor, numtracers;
    int iempty = 0;


    /* XXX: currently only two composition is supported */
    if (E->trace.nflavors != 2) {
        fprintf(E->trace.fpt, "Sorry - Only two flavors of tracers is supported\n");
        fflush(E->trace.fpt);
        parallel_process_termination();
    }


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
                continue;
            }

            /* XXX: generalize for more than one composition */
            flavor = 1;
            E->composition.comp_el[j][e] =
                E->trace.ntracer_flavor[j][flavor][e] / (double)numtracers;
        }


        if (iempty) {

            if ((1.0*iempty/E->lmesh.nel)>0.80) {
                fprintf(E->trace.fpt,"WARNING(compute_elemental...)-number of tracers is REALLY LOW\n");
                fflush(E->trace.fpt);
                if (E->trace.itracer_warnings==1) exit(10);
            }
        }

    } /* end j */

    E->trace.istat_iempty += iempty;

    return;
}

/********** MAP COMPOSITION TO NODES ****************/
/*                                                  */


static void map_composition_to_nodes(struct All_variables *E)
{

    int kk;
    int nelem, nodenum;
    int j;


    for (j=1;j<=E->sphere.caps_per_proc;j++) {

        /* first, initialize node array */
        for (kk=1;kk<=E->lmesh.nno;kk++)
            E->composition.comp_node[j][kk]=0.0;

        /* Loop through all elements */
        for (nelem=1;nelem<=E->lmesh.nel;nelem++) {

            /* for each element, loop through element nodes */

            /* weight composition */

            for (nodenum=1;nodenum<=8;nodenum++) {

                E->composition.comp_node[j][E->ien[j][nelem].node[nodenum]] +=
                    E->composition.comp_el[j][nelem]*
                    E->TWW[E->mesh.levmax][j][nelem].node[nodenum];

            }

        } /* end nelem */
    } /* end j */


    (E->exchange_node_d)(E,E->composition.comp_node,E->mesh.levmax);


    /* Divide by nodal volume */
    for (j=1;j<=E->sphere.caps_per_proc;j++) {
        for (kk=1;kk<=E->lmesh.nno;kk++)
            E->composition.comp_node[j][kk] *= E->MASS[E->mesh.levmax][j][kk];

        /* testing */
        /**
        for (kk=1;kk<=E->lmesh.nel;kk++) {
            fprintf(E->trace.fpt,"%d %f\n",kk,E->composition.comp_el[j][kk]);
        }

        for (kk=1;kk<=E->lmesh.nno;kk++) {
            fprintf(E->trace.fpt,"%d %f %f\n",kk,E->sx[j][3][kk],E->composition.comp_node[j][kk]);
        }
        fflush(E->trace.fpt);
        /**/

    } /* end j */

    return;
}


/*********** GET BULK COMPOSITION *******************************/

static void init_bulk_composition(struct All_variables *E)
{

    char output_file[200];
    char input_s[1000];

    double return_bulk_value_d();
    double volume;
    double rdum1;
    double rdum2;
    double rdum3;

    int ival=0;
    int idum0, idum1;


    FILE *fp;


    /* ival=0 returns integral not average */

    volume = return_bulk_value_d(E,E->composition.comp_node,ival);

    E->composition.bulk_composition = volume;
    E->composition.initial_bulk_composition = volume;


    /* If retarting tracers, the initital bulk composition is read from file */
    // XXX: remove
    if (E->trace.ic_method == 2 &&
        !E->composition.ireset_initial_composition) {

        sprintf(output_file,"%s.comp_el.%d.%d",E->control.old_P_file,
                E->parallel.me, E->monitor.solution_cycles);

        fp=fopen(output_file,"r");
        fgets(input_s,200,fp);
        sscanf(input_s,"%d %d %lf %lf %lf",
               &idum0,&idum1,&rdum1,&rdum2,&rdum3);

        E->composition.initial_bulk_composition = rdum2;
        fclose(fp);

    }

    return;
}


void get_bulk_composition(struct All_variables *E)
{

    double return_bulk_value_d();
    double volume;
    const int ival = 0;

    /* ival=0 returns integral not average */
    volume=return_bulk_value_d(E,E->composition.comp_node,ival);

    E->composition.bulk_composition=volume;

    E->composition.error_fraction=((volume-E->composition.initial_bulk_composition)/
                             E->composition.initial_bulk_composition);

    return;
}
