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
/*

  Tracer_setup.c

      A program which initiates the distribution of tracers
      and advects those tracers in a time evolving velocity field.
      Called and used from the CitCOM finite element code.
      Written 2/96 M. Gurnis for Citcom in cartesian geometry
      Modified by Lijie in 1998 and by Vlad and Eh in 2005 for the
      regional version of CitcomS. In 2003, Allen McNamara wrote the
      tracer module for the global version of CitcomS. In 2007, Eh Tan
      merged the two versions of tracer codes together.
*/

#include <math.h>
#include "global_defs.h"
#include "parsing.h"
#include "parallel_related.h"
#include "composition_related.h"

#ifdef USE_GGRD
#include "ggrd_handling.h"
#endif

#ifdef USE_GZDIR
int open_file_zipped(char *, FILE **,struct All_variables *);
void gzip_file(char *);
#endif

int icheck_that_processor_shell(struct All_variables *E,
                                       int j, int nprocessor, double rad);
void expand_later_array(struct All_variables *E, int j);
void expand_tracer_arrays(struct All_variables *E, int j);
void tracer_post_processing(struct All_variables *E);
void allocate_tracer_arrays(struct All_variables *E, int number_of_tracers);
void count_tracers_of_flavors(struct All_variables *E);

int full_icheck_cap(struct All_variables *E, int icap,
                    double x, double y, double z, double rad);
int regional_icheck_cap(struct All_variables *E, int icap,
                        double x, double y, double z, double rad);

static void find_tracers(struct All_variables *E);
static void predict_tracers(struct All_variables *E);
static void correct_tracers(struct All_variables *E);
static void make_tracer_array(struct All_variables *E);
static void generate_random_tracers(struct All_variables *E,
                                    int tracers_cap, int j);
static void read_tracer_file(struct All_variables *E);
static void read_old_tracer_file(struct All_variables *E);
static void check_sum(struct All_variables *E);
static int isum_tracers(struct All_variables *E);
static void init_tracer_flavors(struct All_variables *E);
static void reduce_tracer_arrays(struct All_variables *E);
static void put_away_later(struct All_variables *E, int j, int it);
static void eject_tracer(struct All_variables *E, int it);
int read_double_vector(FILE *, int , double *);
void cart_to_sphere(struct All_variables *,
                    double , double , double ,
                    double *, double *, double *);
void sphere_to_cart(struct All_variables *,
                    double , double , double ,
                    double *, double *, double *);
int icheck_processor_shell(struct All_variables *,
                           int , double );



void tracer_input(struct All_variables *E)
{
    void full_tracer_input();
    void myerror();
    void report();
    char message[100];
    int m=E->parallel.me;
    int i;

    input_boolean("tracer",&(E->control.tracer),"off",m);
    input_boolean("tracer_enriched",
		  &(E->control.tracer_enriched),"off",m);
    if(E->control.tracer_enriched){
      if(!E->control.tracer)	/* check here so that we can get away
				   with only one if statement in
				   Advection_diffusion */
	myerror(E,"need to switch on tracers for tracer_enriched");

      input_float("Q0_enriched",&(E->control.Q0ER),"0.0",m);
      snprintf(message,100,"using compositionally enriched heating: C = 0: %g C = 1: %g (only one composition!)",
	       E->control.Q0,E->control.Q0ER);
      report(E,message);
      //
      // this check doesn't work at this point in the code, and we didn't want to put it into every call to
      // Advection diffusion
      //
      //if(E->composition.ncomp != 1)
      //myerror(E,"enriched tracers cannot deal with more than one composition yet");

    }
    if(E->control.tracer) {

        /* tracer_ic_method=0 (random generated array) */
        /* tracer_ic_method=1 (all proc read the same file) */
        /* tracer_ic_method=2 (each proc reads its restart file) */
        input_int("tracer_ic_method",&(E->trace.ic_method),"0,0,nomax",m);

        if (E->trace.ic_method==0){
            input_int("tracers_per_element",&(E->trace.itperel),"10,0,nomax",m);
	}
        else if (E->trace.ic_method==1)
            input_string("tracer_file",E->trace.tracer_file,"tracer.dat",m);
        else if (E->trace.ic_method==2) {
            /* Use 'datadir_old', 'datafile_old', and 'solution_cycles_init' */
            /* to form the filename */
        }
        else {
            fprintf(stderr,"Sorry, tracer_ic_method only 0, 1 and 2 available\n");
            parallel_process_termination();
        }


        /* How many flavors of tracers */
        /* If tracer_flavors > 0, each element will report the number of
         * tracers of each flavor inside it. This information can be used
         * later for many purposes. One of it is to compute composition,
         * either using absolute method or ratio method. */
        input_int("tracer_flavors",&(E->trace.nflavors),"0,0,nomax",m);

	/* 0: default from layers 
	   1: from netcdf grds
	   
	   
	   99: from grds, overriding checkpoints during restart
	   (1 and 99 require ggrd)
	*/

        input_int("ic_method_for_flavors",
		  &(E->trace.ic_method_for_flavors),"0,0,nomax",m);


        if (E->trace.nflavors > 1) {
            switch(E->trace.ic_method_for_flavors){
	      /* default method */
            case 0:			
	      /* flavors initialized from layers */
                E->trace.z_interface = (double*) malloc((E->trace.nflavors-1)
                                                        *sizeof(double));
                for(i=0; i<E->trace.nflavors-1; i++)
                    E->trace.z_interface[i] = 0.7;

                input_double_vector("z_interface", E->trace.nflavors-1,
                                    E->trace.z_interface, m);
                break;
		/* 
		   two grd init method, second will override restart
		*/
#ifdef USE_GGRD
            case 1:
	    case 99:		/* will override restart */
	      /* from grid in top n materials, this will override
		 the checkpoint input */
	      input_string("ictracer_grd_file",E->trace.ggrd_file,"",m); /* file from which to read */
	      input_int("ictracer_grd_layers",&(E->trace.ggrd_layers),"2",m); /* 

									      >0 : which top layers to use, layer <= ictracer_grd_layers
									      <0 : only use one layer layer == -ictracer_grd_layers

									      */
	      break;
	      
#endif
            default:
                fprintf(stderr,"ic_method_for_flavors %i undefined (1 and 99 only for ggrd mode)\n",E->trace.ic_method_for_flavors);
                parallel_process_termination();
                break;
            }
        }

        /* Warning level */
        input_boolean("itracer_warnings",&(E->trace.itracer_warnings),"on",m);


        if(E->parallel.nprocxy == 12)
            full_tracer_input(E);


        composition_input(E);

    }

    return;
}


void tracer_initial_settings(struct All_variables *E)
{
   void full_keep_within_bounds();
   void full_tracer_setup();
   void full_get_velocity();
   int full_iget_element();
   void regional_keep_within_bounds();
   void regional_tracer_setup();
   void regional_get_velocity();
   int regional_iget_element();

   E->trace.advection_time = 0;
   E->trace.find_tracers_time = 0;
   E->trace.lost_souls_time = 0;

   if(E->parallel.nprocxy == 1) {
       E->problem_tracer_setup = regional_tracer_setup;

       E->trace.keep_within_bounds = regional_keep_within_bounds;
       E->trace.get_velocity = regional_get_velocity;
       E->trace.iget_element = regional_iget_element;
   }
   else {
       E->problem_tracer_setup = full_tracer_setup;

       E->trace.keep_within_bounds = full_keep_within_bounds;
       E->trace.get_velocity = full_get_velocity;
       E->trace.iget_element = full_iget_element;
   }
}



/*****************************************************************************/
/* This function is the primary tracing routine called from Citcom.c         */
/* In this code, unlike the original 3D cartesian code, force is filled      */
/* during Stokes solution. No need to call thermal_buoyancy() after tracing. */


void tracer_advection(struct All_variables *E)
{
    double CPU_time0();
    double begin_time = CPU_time0();

    /* advect tracers */
    predict_tracers(E);
    correct_tracers(E);

    /* check that the number of tracers is conserved */
    check_sum(E);

    /* count # of tracers of each flavor */
    if (E->trace.nflavors > 0)
        count_tracers_of_flavors(E);

    /* update the composition field */
    if (E->composition.on) {
        fill_composition(E);
    }

    E->trace.advection_time += CPU_time0() - begin_time;

    tracer_post_processing(E);

    return;
}



/********* TRACER POST PROCESSING ****************************************/

void tracer_post_processing(struct All_variables *E)
{
    int i;

    /* reset statistical counters */

    E->trace.istat_isend=0;
    E->trace.istat_elements_checked=0;
    E->trace.istat1=0;

    /* write timing information every 20 steps */
    if ((E->monitor.solution_cycles % 20) == 0) {
        fprintf(E->trace.fpt, "STEP %d\n", E->monitor.solution_cycles);

        fprintf(E->trace.fpt, "Advecting tracers takes %f seconds.\n",
                E->trace.advection_time - E->trace.find_tracers_time);
        fprintf(E->trace.fpt, "Finding element takes %f seconds.\n",
                E->trace.find_tracers_time - E->trace.lost_souls_time);
        fprintf(E->trace.fpt, "Exchanging lost tracers takes %f seconds.\n",
                E->trace.lost_souls_time);
    }

    if(E->control.verbose){
      fprintf(E->trace.fpt,"Number of times for all element search  %d\n",E->trace.istat1);

      fprintf(E->trace.fpt,"Number of tracers sent to other processors: %d\n",E->trace.istat_isend);

      fprintf(E->trace.fpt,"Number of times element columns are checked: %d \n",E->trace.istat_elements_checked);

      /* compositional and error fraction data files */
      //TODO: move
      if (E->composition.on) {
        fprintf(E->trace.fpt,"Empty elements filled with old compositional "
                "values: %d (%f percent)\n", E->trace.istat_iempty,
                (100.0*E->trace.istat_iempty)/E->lmesh.nel);
        E->trace.istat_iempty=0;


        get_bulk_composition(E);

        if (E->parallel.me==0) {

            fprintf(E->fp,"composition: %e",E->monitor.elapsed_time);
            for (i=0; i<E->composition.ncomp; i++)
                fprintf(E->fp," %e", E->composition.bulk_composition[i]);
            fprintf(E->fp,"\n");

            fprintf(E->fp,"composition_error_fraction: %e",E->monitor.elapsed_time);
            for (i=0; i<E->composition.ncomp; i++)
                fprintf(E->fp," %e", E->composition.error_fraction[i]);
            fprintf(E->fp,"\n");

        }
      }
      fflush(E->trace.fpt);
    }

    return;
}


/*********** PREDICT TRACERS **********************************************/
/*                                                                        */
/* This function predicts tracers performing an euler step                */
/*                                                                        */
/*                                                                        */
/* Note positions used in tracer array                                    */
/* [positions 0-5 are always fixed with current coordinates               */
/*  Positions 6-8 contain original Cartesian coordinates.                 */
/*  Positions 9-11 contain original Cartesian velocities.                 */
/*                                                                        */


static void predict_tracers(struct All_variables *E)
{

    int numtracers;
    int j;
    int kk;
    int nelem;

    double dt;
    double theta0,phi0,rad0;
    double x0,y0,z0;
    double theta_pred,phi_pred,rad_pred;
    double x_pred,y_pred,z_pred;
    double velocity_vector[4];

    void cart_to_sphere();


    dt=E->advection.timestep;



        numtracers=E->trace.ntracers[CPPR];

        for (kk=1;kk<=numtracers;kk++) {

            theta0=E->trace.basicq[CPPR][0][kk];
            phi0=E->trace.basicq[CPPR][1][kk];
            rad0=E->trace.basicq[CPPR][2][kk];
            x0=E->trace.basicq[CPPR][3][kk];
            y0=E->trace.basicq[CPPR][4][kk];
            z0=E->trace.basicq[CPPR][5][kk];

            nelem=E->trace.ielement[CPPR][kk];
            (E->trace.get_velocity)(E,CPPR,nelem,theta0,phi0,rad0,velocity_vector);

            x_pred=x0+velocity_vector[1]*dt;
            y_pred=y0+velocity_vector[2]*dt;
            z_pred=z0+velocity_vector[3]*dt;


            /* keep in box */

            cart_to_sphere(E,x_pred,y_pred,z_pred,&theta_pred,&phi_pred,&rad_pred);
            (E->trace.keep_within_bounds)(E,&x_pred,&y_pred,&z_pred,&theta_pred,&phi_pred,&rad_pred);

            /* Current Coordinates are always kept in positions 0-5. */

            E->trace.basicq[CPPR][0][kk]=theta_pred;
            E->trace.basicq[CPPR][1][kk]=phi_pred;
            E->trace.basicq[CPPR][2][kk]=rad_pred;
            E->trace.basicq[CPPR][3][kk]=x_pred;
            E->trace.basicq[CPPR][4][kk]=y_pred;
            E->trace.basicq[CPPR][5][kk]=z_pred;

            /* Fill in original coords (positions 6-8) */

            E->trace.basicq[CPPR][6][kk]=x0;
            E->trace.basicq[CPPR][7][kk]=y0;
            E->trace.basicq[CPPR][8][kk]=z0;

            /* Fill in original velocities (positions 9-11) */

            E->trace.basicq[CPPR][9][kk]=velocity_vector[1];  /* Vx */
            E->trace.basicq[CPPR][10][kk]=velocity_vector[2];  /* Vy */
            E->trace.basicq[CPPR][11][kk]=velocity_vector[3];  /* Vz */


        } /* end kk, predicting tracers */

    /* find new tracer elements and caps */

    find_tracers(E);

    return;

}


/*********** CORRECT TRACERS **********************************************/
/*                                                                        */
/* This function corrects tracers using both initial and                  */
/* predicted velocities                                                   */
/*                                                                        */
/*                                                                        */
/* Note positions used in tracer array                                    */
/* [positions 0-5 are always fixed with current coordinates               */
/*  Positions 6-8 contain original Cartesian coordinates.                 */
/*  Positions 9-11 contain original Cartesian velocities.                 */
/*                                                                        */


static void correct_tracers(struct All_variables *E)
{

    int j;
    int kk;
    int nelem;


    double dt;
    double x0,y0,z0;
    double theta_pred,phi_pred,rad_pred;
    double x_pred,y_pred,z_pred;
    double theta_cor,phi_cor,rad_cor;
    double x_cor,y_cor,z_cor;
    double velocity_vector[4];
    double Vx0,Vy0,Vz0;
    double Vx_pred,Vy_pred,Vz_pred;

    void cart_to_sphere();


    dt=E->advection.timestep;


        for (kk=1;kk<=E->trace.ntracers[CPPR];kk++) {

            theta_pred=E->trace.basicq[CPPR][0][kk];
            phi_pred=E->trace.basicq[CPPR][1][kk];
            rad_pred=E->trace.basicq[CPPR][2][kk];
            x_pred=E->trace.basicq[CPPR][3][kk];
            y_pred=E->trace.basicq[CPPR][4][kk];
            z_pred=E->trace.basicq[CPPR][5][kk];

            x0=E->trace.basicq[CPPR][6][kk];
            y0=E->trace.basicq[CPPR][7][kk];
            z0=E->trace.basicq[CPPR][8][kk];

            Vx0=E->trace.basicq[CPPR][9][kk];
            Vy0=E->trace.basicq[CPPR][10][kk];
            Vz0=E->trace.basicq[CPPR][11][kk];

            nelem=E->trace.ielement[CPPR][kk];

            (E->trace.get_velocity)(E,CPPR,nelem,theta_pred,phi_pred,rad_pred,velocity_vector);

            Vx_pred=velocity_vector[1];
            Vy_pred=velocity_vector[2];
            Vz_pred=velocity_vector[3];

            x_cor=x0 + dt * 0.5*(Vx0+Vx_pred);
            y_cor=y0 + dt * 0.5*(Vy0+Vy_pred);
            z_cor=z0 + dt * 0.5*(Vz0+Vz_pred);

            cart_to_sphere(E,x_cor,y_cor,z_cor,&theta_cor,&phi_cor,&rad_cor);
            (E->trace.keep_within_bounds)(E,&x_cor,&y_cor,&z_cor,&theta_cor,&phi_cor,&rad_cor);

            /* Fill in Current Positions (other positions are no longer important) */

            E->trace.basicq[CPPR][0][kk]=theta_cor;
            E->trace.basicq[CPPR][1][kk]=phi_cor;
            E->trace.basicq[CPPR][2][kk]=rad_cor;
            E->trace.basicq[CPPR][3][kk]=x_cor;
            E->trace.basicq[CPPR][4][kk]=y_cor;
            E->trace.basicq[CPPR][5][kk]=z_cor;

        } /* end kk, correcting tracers */

    /* find new tracer elements and caps */

    find_tracers(E);

    return;
}


/************ FIND TRACERS *************************************/
/*                                                             */
/* This function finds tracer elements and moves tracers to    */
/* other processor domains if necessary.                       */
/* Array ielement is filled with elemental values.                */

static void find_tracers(struct All_variables *E)
{

    int iel;
    int kk;
    int j;
    int it;
    int iprevious_element;
    int num_tracers;

    double theta,phi,rad;
    double x,y,z;
    double time_stat1;
    double time_stat2;

    void put_away_later();
    void eject_tracer();
    void reduce_tracer_arrays();
    void sphere_to_cart();
    void full_lost_souls();
    void regional_lost_souls();

    double CPU_time0();
    double begin_time = CPU_time0();




        /* initialize arrays and statistical counters */

        E->trace.ilater[CPPR]=E->trace.ilatersize[CPPR]=0;

        E->trace.istat1=0;
        for (kk=0;kk<=4;kk++) {
            E->trace.istat_ichoice[CPPR][kk]=0;
        }

        //TODO: use while-loop instead of for-loop
        /* important to index by it, not kk */

        it=0;
        num_tracers=E->trace.ntracers[CPPR];

        for (kk=1;kk<=num_tracers;kk++) {

            it++;

            theta=E->trace.basicq[CPPR][0][it];
            phi=E->trace.basicq[CPPR][1][it];
            rad=E->trace.basicq[CPPR][2][it];
            x=E->trace.basicq[CPPR][3][it];
            y=E->trace.basicq[CPPR][4][it];
            z=E->trace.basicq[CPPR][5][it];

            iprevious_element=E->trace.ielement[CPPR][it];

            iel=(E->trace.iget_element)(E,CPPR,iprevious_element,x,y,z,theta,phi,rad);

            E->trace.ielement[CPPR][it]=iel;

            if (iel == -99) {
                /* tracer is inside other processors */
                put_away_later(E,CPPR,it);
                eject_tracer(E,it);
                it--;
            } else if (iel == -1) {
                /* tracer is inside this processor,
                 * but cannot find its element.
                 * Throw away the tracer. */

                if (E->trace.itracer_warnings) exit(10);


                eject_tracer(E,it);
                it--;
            }

        } /* end tracers */



    /* Now take care of tracers that exited cap */

    /* REMOVE */
    /*
      parallel_process_termination();
    */

    if (E->parallel.nprocxy == 12)
        full_lost_souls(E);
    else
        regional_lost_souls(E);

    /* Free later arrays */

    if (E->trace.ilatersize[CPPR]>0) {
        for (kk=0;kk<=((E->trace.number_of_tracer_quantities)-1);kk++) {
            free(E->trace.rlater[CPPR][kk]);
        }
    }


    /* Adjust Array Sizes */

    reduce_tracer_arrays(E);

    E->trace.find_tracers_time += CPU_time0() - begin_time;

    return;
}


/***********************************************************************/
/* This function computes the number of tracers in each element.       */
/* Each tracer can be of different "flavors", which is the 0th index   */
/* of extraq. How to interprete "flavor" is left for the application.  */

void count_tracers_of_flavors(struct All_variables *E)
{

    int j, flavor, e, kk;
    int numtracers;


        /* first zero arrays */
        for (flavor=0; flavor<E->trace.nflavors; flavor++)
            for (e=1; e<=E->lmesh.nel; e++)
                E->trace.ntracer_flavor[CPPR][flavor][e] = 0;

        numtracers=E->trace.ntracers[CPPR];

        /* Fill arrays */
        for (kk=1; kk<=numtracers; kk++) {
            e = E->trace.ielement[CPPR][kk];
            flavor = E->trace.extraq[CPPR][0][kk];
            E->trace.ntracer_flavor[CPPR][flavor][e]++;
        }
}



void initialize_tracers(struct All_variables *E)
{

    if (E->trace.ic_method==0)
        make_tracer_array(E);
    else if (E->trace.ic_method==1)
        read_tracer_file(E);
    else if (E->trace.ic_method==2)
        read_old_tracer_file(E);
    else {
        fprintf(E->trace.fpt,"Not ready for other inputs yet\n");
        fflush(E->trace.fpt);
        parallel_process_termination();
    }

    /* total number of tracers  */

    E->trace.ilast_tracer_count = isum_tracers(E);
    fprintf(E->trace.fpt, "Sum of Tracers: %d\n", E->trace.ilast_tracer_count);
    if(E->parallel.me==0)
        fprintf(stderr, "Sum of Tracers: %d\n", E->trace.ilast_tracer_count);


    /* find elements */

    find_tracers(E);


    /* count # of tracers of each flavor */

    if (E->trace.nflavors > 0)
        count_tracers_of_flavors(E);

    return;
}


/************** MAKE TRACER ARRAY ********************************/
/* Here, each processor will generate tracers somewhere          */
/* in the sphere - check if its in this cap  - then check radial */

static void make_tracer_array(struct All_variables *E)
{

    int tracers_cap;
    int j;
    double processor_fraction;

    void generate_random_tracers();
    void init_tracer_flavors();

    if (E->parallel.me==0) fprintf(stderr,"Making Tracer Array\n");

        processor_fraction=E->lmesh.volume/E->mesh.volume;
        tracers_cap=E->mesh.nel*E->trace.itperel*processor_fraction;
        /*
          fprintf(stderr,"AA: proc frac: %f (%d) %d %d %f %f\n",processor_fraction,tracers_cap,E->lmesh.nel,E->parallel.nprocz, E->sx[j][3][E->lmesh.noz],E->sx[j][3][1]);
        */

        fprintf(E->trace.fpt,"\nGenerating %d Tracers\n",tracers_cap);

        generate_random_tracers(E, tracers_cap, CPPR);





    /* Initialize tracer flavors */
    if (E->trace.nflavors) init_tracer_flavors(E);
}



static void generate_random_tracers(struct All_variables *E,
                                    int tracers_cap, int j)
{
    void cart_to_sphere();
    int kk;
    int ival;
    int number_of_tries=0;
    int max_tries;

    double x,y,z;
    double theta,phi,rad;
    double xmin,xmax,ymin,ymax,zmin,zmax;
    double random1,random2,random3;


    allocate_tracer_arrays(E,tracers_cap);

    /* Finding the min/max of the cartesian coordinates. */
    /* One must loop over E->X to find the min/max, since the 8 corner */
    /* nodes may not be the min/max. */
    xmin = ymin = zmin = E->sphere.ro;
    xmax = ymax = zmax = -E->sphere.ro;
    for (kk=1; kk<=E->lmesh.nno; kk++) {
        x = E->x[CPPR][1][kk];
        y = E->x[CPPR][2][kk];
        z = E->x[CPPR][3][kk];

        xmin = ((xmin < x) ? xmin : x);
        xmax = ((xmax > x) ? xmax : x);
        ymin = ((ymin < y) ? ymin : y);
        ymax = ((ymax > y) ? ymax : y);
        zmin = ((zmin < z) ? zmin : z);
        zmax = ((zmax > z) ? zmax : z);
    }

    /* Tracers are placed randomly in cap */
    /* (intentionally using rand() instead of srand() )*/
    while (E->trace.ntracers[CPPR]<tracers_cap) {

        number_of_tries++;
        max_tries=100*tracers_cap;

        if (number_of_tries>max_tries) {
            fprintf(E->trace.fpt,"Error(make_tracer_array)-too many tries?\n");
            fprintf(E->trace.fpt,"%d %d %d\n",max_tries,number_of_tries,RAND_MAX);
            fflush(E->trace.fpt);
            exit(10);
        }

#if 1
        random1=drand48();
        random2=drand48();
        random3=drand48();
#else  /* never called */
        random1=(1.0*rand())/(1.0*RAND_MAX);
        random2=(1.0*rand())/(1.0*RAND_MAX);
        random3=(1.0*rand())/(1.0*RAND_MAX);
#endif

        x=xmin+random1*(xmax-xmin);
        y=ymin+random2*(ymax-ymin);
        z=zmin+random3*(zmax-zmin);

        /* first check if within shell */

        cart_to_sphere(E,x,y,z,&theta,&phi,&rad);

        if (rad>=E->sx[CPPR][3][E->lmesh.noz]) continue;
        if (rad<E->sx[CPPR][3][1]) continue;


        /* check if in current cap */
        if (E->parallel.nprocxy==1)
            ival=regional_icheck_cap(E,0,theta,phi,rad,rad);
        else
            ival=full_icheck_cap(E,0,x,y,z,rad);

        if (ival!=1) continue;

        /* Made it, so record tracer information */

        (E->trace.keep_within_bounds)(E,&x,&y,&z,&theta,&phi,&rad);

        E->trace.ntracers[CPPR]++;
        kk=E->trace.ntracers[CPPR];

        E->trace.basicq[CPPR][0][kk]=theta;
        E->trace.basicq[CPPR][1][kk]=phi;
        E->trace.basicq[CPPR][2][kk]=rad;
        E->trace.basicq[CPPR][3][kk]=x;
        E->trace.basicq[CPPR][4][kk]=y;
        E->trace.basicq[CPPR][5][kk]=z;

    } /* end while */

    return;
}


/******** READ TRACER ARRAY *********************************************/
/*                                                                      */
/* This function reads tracers from input file.                         */
/* All processors read the same input file, then sort out which ones    */
/* belong.                                                              */

static void read_tracer_file(struct All_variables *E)
{

    char input_s[1000];

    int number_of_tracers, ncolumns;
    int kk;
    int icheck;
    int iestimate;
    int icushion;
    int i, j;


    int icheck_processor_shell();
    void sphere_to_cart();
    void cart_to_sphere();
    void expand_tracer_arrays();

    double x,y,z;
    double theta,phi,rad;
    double buffer[100];

    FILE *fptracer;

    fptracer=fopen(E->trace.tracer_file,"r");

    fgets(input_s,200,fptracer);
    if(sscanf(input_s,"%d %d",&number_of_tracers,&ncolumns) != 2) {
        fprintf(stderr,"Error while reading file '%s'\n", E->trace.tracer_file);
        exit(8);
    }
    fprintf(E->trace.fpt,"%d Tracers, %d columns in file \n",
            number_of_tracers, ncolumns);

    /* some error control */
    if (E->trace.number_of_extra_quantities+3 != ncolumns) {
        fprintf(E->trace.fpt,"ERROR(read tracer file)-wrong # of columns\n");
        fflush(E->trace.fpt);
        exit(10);
    }


    /* initially size tracer arrays to number of tracers divided by processors */

    icushion=100;

    /* for absolute tracer method */
    E->trace.number_of_tracers = number_of_tracers;

    iestimate=number_of_tracers/E->parallel.nproc + icushion;

        allocate_tracer_arrays(E,iestimate);

        for (kk=1;kk<=number_of_tracers;kk++) {
            int len, ncol;
            ncol = 3 + E->trace.number_of_extra_quantities;

            len = read_double_vector(fptracer, ncol, buffer);
            if (len != ncol) {
                fprintf(E->trace.fpt,"ERROR(read tracer file) - wrong input file format: %s\n", E->trace.tracer_file);
                fflush(E->trace.fpt);
                exit(10);
            }

            theta = buffer[0];
            phi = buffer[1];
            rad = buffer[2];

            sphere_to_cart(E,theta,phi,rad,&x,&y,&z);


            /* make sure theta, phi is in range, and radius is within bounds */

            (E->trace.keep_within_bounds)(E,&x,&y,&z,&theta,&phi,&rad);

            /* check whether tracer is within processor domain */

            icheck=1;
            if (E->parallel.nprocz>1) icheck=icheck_processor_shell(E,CPPR,rad);
            if (icheck!=1) continue;

            if (E->parallel.nprocxy==1)
                icheck=regional_icheck_cap(E,0,theta,phi,rad,rad);
            else
                icheck=full_icheck_cap(E,0,x,y,z,rad);

            if (icheck==0) continue;

            /* if still here, tracer is in processor domain */


            E->trace.ntracers[CPPR]++;

            if (E->trace.ntracers[CPPR]>=(E->trace.max_ntracers[CPPR]-5)) expand_tracer_arrays(E,j);

            E->trace.basicq[CPPR][0][E->trace.ntracers[CPPR]]=theta;
            E->trace.basicq[CPPR][1][E->trace.ntracers[CPPR]]=phi;
            E->trace.basicq[CPPR][2][E->trace.ntracers[CPPR]]=rad;
            E->trace.basicq[CPPR][3][E->trace.ntracers[CPPR]]=x;
            E->trace.basicq[CPPR][4][E->trace.ntracers[CPPR]]=y;
            E->trace.basicq[CPPR][5][E->trace.ntracers[CPPR]]=z;

            for (i=0; i<E->trace.number_of_extra_quantities; i++)
                E->trace.extraq[CPPR][i][E->trace.ntracers[CPPR]]=buffer[i+3];

        } /* end kk, number of tracers */

        fprintf(E->trace.fpt,"Number of tracers in this cap is: %d\n",
                E->trace.ntracers[CPPR]);

    fclose(fptracer);

    icheck=isum_tracers(E);

    if (icheck!=number_of_tracers) {
        fprintf(E->trace.fpt,"ERROR(read_tracer_file) - tracers != number in file\n");
        fprintf(E->trace.fpt,"Tracers in system: %d\n", icheck);
        fprintf(E->trace.fpt,"Tracers in file: %d\n", number_of_tracers);
        fflush(E->trace.fpt);
        exit(10);
    }

    return;
}


/************** READ OLD TRACER FILE *************************************/
/*                                                                       */
/* This function read tracers written from previous calculation          */
/* and the tracers are read as seperate files for each processor domain. */

static void read_old_tracer_file(struct All_variables *E)
{

    char output_file[200];
    char input_s[1000];

    int i,j,kk,rezip;
    int idum1,ncolumns;
    int numtracers;

    double rdum1;
    double theta,phi,rad;
    double x,y,z;
    double buffer[100];

    void sphere_to_cart();

    FILE *fp1;

    if (E->trace.number_of_extra_quantities>99) {
        fprintf(E->trace.fpt,"ERROR(read_old_tracer_file)-increase size of extra[]\n");
        fflush(E->trace.fpt);
        parallel_process_termination();
    }



    /* deal with different output formats */
#ifdef USE_GZDIR
    if(strcmp(E->output.format, "ascii-gz") == 0){
      sprintf(output_file,"%s/%d/tracer.%d.%d",
	      E->control.data_dir_old,E->monitor.solution_cycles_init,E->parallel.me,E->monitor.solution_cycles_init);
      rezip = open_file_zipped(output_file,&fp1,E);
    }else{
      sprintf(output_file,"%s.tracer.%d.%d",E->control.old_P_file,E->parallel.me,E->monitor.solution_cycles_init);
      if ( (fp1=fopen(output_file,"r"))==NULL) {
        fprintf(E->trace.fpt,"ERROR(read_old_tracer_file)-gziped file not found %s\n",output_file);
        fflush(E->trace.fpt);
        exit(10);
      }
    }
#else
    sprintf(output_file,"%s.tracer.%d.%d",E->control.old_P_file,E->parallel.me,E->monitor.solution_cycles_init);
    if ( (fp1=fopen(output_file,"r"))==NULL) {
        fprintf(E->trace.fpt,"ERROR(read_old_tracer_file)-file not found %s\n",output_file);
        fflush(E->trace.fpt);
        exit(10);
    }
#endif

    fprintf(stderr,"Read old tracers from %s\n",output_file);


        fgets(input_s,200,fp1);
        if(sscanf(input_s,"%d %d %d %lf",
                  &idum1, &numtracers, &ncolumns, &rdum1) != 4) {
            fprintf(stderr,"Error while reading file '%s'\n", output_file);
            exit(8);
        }


        /* some error control */
        if (E->trace.number_of_extra_quantities+3 != ncolumns) {
            fprintf(E->trace.fpt,"ERROR(read_old_tracer_file)-wrong # of columns\n");
            fflush(E->trace.fpt);
            exit(10);
        }

        /* allocate memory for tracer arrays */

        allocate_tracer_arrays(E,numtracers);
        E->trace.ntracers[CPPR]=numtracers;

        for (kk=1;kk<=numtracers;kk++) {
            int len, ncol;
            ncol = 3 + E->trace.number_of_extra_quantities;

            len = read_double_vector(fp1, ncol, buffer);
            if (len != ncol) {
                fprintf(E->trace.fpt,"ERROR(read_old_tracer_file) - wrong input file format: %s\n", output_file);
                fflush(E->trace.fpt);
                exit(10);
            }

            theta = buffer[0];
            phi = buffer[1];
            rad = buffer[2];

            sphere_to_cart(E,theta,phi,rad,&x,&y,&z);

            /* it is possible that if on phi=0 boundary, significant digits can push phi over 2pi */

            (E->trace.keep_within_bounds)(E,&x,&y,&z,&theta,&phi,&rad);

            E->trace.basicq[CPPR][0][kk]=theta;
            E->trace.basicq[CPPR][1][kk]=phi;
            E->trace.basicq[CPPR][2][kk]=rad;
            E->trace.basicq[CPPR][3][kk]=x;
            E->trace.basicq[CPPR][4][kk]=y;
            E->trace.basicq[CPPR][5][kk]=z;

            for (i=0; i<E->trace.number_of_extra_quantities; i++)
                E->trace.extraq[CPPR][i][kk]=buffer[i+3];

        }

        fprintf(E->trace.fpt,"Read %d tracers from file %s\n",numtracers,output_file);
        fflush(E->trace.fpt);

    fclose(fp1);
#ifdef USE_GZDIR
    if(strcmp(E->output.format, "ascii-gz") == 0)
      if(rezip)			/* rezip */
	gzip_file(output_file);
#endif

    return;
}





/*********** CHECK SUM **************************************************/
/*                                                                      */
/* This functions checks to make sure number of tracers is preserved    */

static void check_sum(struct All_variables *E)
{

    int number, iold_number;

    number = isum_tracers(E);

    iold_number = E->trace.ilast_tracer_count;

    if (number != iold_number) {
        fprintf(E->trace.fpt,"ERROR(check_sum)-break in conservation %d %d\n",
                number,iold_number);
        fflush(E->trace.fpt);
        if (E->trace.itracer_warnings)
            parallel_process_termination();
    }

    E->trace.ilast_tracer_count = number;

    return;
}


/************* ISUM TRACERS **********************************************/
/*                                                                       */
/* This function uses MPI to sum all tracers and returns number of them. */

static int isum_tracers(struct All_variables *E)
{
    int imycount;
    int iallcount;
    int j;

    iallcount = 0;

    imycount = 0;
    imycount = imycount + E->trace.ntracers[CPPR];

    MPI_Allreduce(&imycount,&iallcount,1,MPI_INT,MPI_SUM,E->parallel.world);

    return iallcount;
}



/********** CART TO SPHERE ***********************/
void cart_to_sphere(struct All_variables *E,
                    double x, double y, double z,
                    double *theta, double *phi, double *rad)
{

    double temp;
    double myatan();

    temp=x*x+y*y;

    *rad=sqrt(temp+z*z);
    *theta=atan2(sqrt(temp),z);
    *phi=myatan(y,x);
}

/********** SPHERE TO CART ***********************/
void sphere_to_cart(struct All_variables *E,
                    double theta, double phi, double rad,
                    double *x, double *y, double *z)
{

    double sint,cost,sinf,cosf;
    double temp;

    sint=sin(theta);
    cost=cos(theta);
    sinf=sin(phi);
    cosf=cos(phi);

    temp=rad*sint;

    *x=temp*cosf;
    *y=temp*sinf;
    *z=rad*cost;

    return;
}



static void init_tracer_flavors(struct All_variables *E)
{
    int j, kk, number_of_tracers;
    int i;
    double flavor;
    double rad;

    switch(E->trace.ic_method_for_flavors){
    case 0:
      /* ic_method_for_flavors == 0 (layered structure) */
      /* any tracer above z_interface[i] is of flavor i */
      /* any tracer below z_interface is of flavor (nflavors-1) */

	number_of_tracers = E->trace.ntracers[CPPR];
	for (kk=1;kk<=number_of_tracers;kk++) {
	  rad = E->trace.basicq[CPPR][2][kk];

          flavor = E->trace.nflavors - 1;
          for (i=0; i<E->trace.nflavors-1; i++) {
              if (rad > E->trace.z_interface[i]) {
                  flavor = i;
                  break;
              }
          }
          E->trace.extraq[CPPR][0][kk] = flavor;
	}
      break;

    case 1:			/* from grd in top n layers */
    case 99:			/* (will override restart) */
#ifndef USE_GGRD
      fprintf(stderr,"ic_method_for_flavors %i requires the ggrd routines from hc, -DUSE_GGRD\n",
	      E->trace.ic_method_for_flavors);
      parallel_process_termination();
#else
      ggrd_init_tracer_flavors(E);
#endif
      break;


    default:

      fprintf(stderr,"ic_method_for_flavors %i undefined\n",E->trace.ic_method_for_flavors);
      parallel_process_termination();
      break;
    }

    return;
}


/******************* get_neighboring_caps ************************************/
/*                                                                           */
/* Communicate with neighboring processors to get their cap boundaries,      */
/* which is later used by (E->trace.icheck_cap)()                            */
/*                                                                           */

void get_neighboring_caps(struct All_variables *E)
{
    void sphere_to_cart();

    const int ncorners = 4; /* # of top corner nodes */
    int i, j, n, d, kk, lev, idb;
    int num_ngb, neighbor_proc, tag;
    MPI_Status status[200];
    MPI_Request request[200];

    int node[ncorners];
    double xx[ncorners*2], rr[12][ncorners*2];
    int nox,noy,noz;
    double x,y,z;
    double theta,phi,rad;

    nox=E->lmesh.nox;
    noy=E->lmesh.noy;
    noz=E->lmesh.noz;

    node[0]=nox*noz*(noy-1)+noz;
    node[1]=noz;
    node[2]=noz*nox;
    node[3]=noz*nox*noy;

    lev = E->mesh.levmax;
    tag = 45;

        /* loop over top corners to get their coordinates */
        n = 0;
        for (i=0; i<ncorners; i++) {
            for (d=0; d<2; d++) {
                xx[n] = E->sx[CPPR][d+1][node[i]];
                n++;
            }
        }

        idb = 0;
        num_ngb = E->parallel.TNUM_PASS[lev][CPPR];
        for (kk=1; kk<=num_ngb; kk++) {
            neighbor_proc = E->parallel.PROCESSOR[lev][CPPR].pass[kk];

            MPI_Isend(xx, n, MPI_DOUBLE, neighbor_proc,
                      tag, E->parallel.world, &request[idb]);
            idb++;

            MPI_Irecv(rr[kk], n, MPI_DOUBLE, neighbor_proc,
                      tag, E->parallel.world, &request[idb]);
            idb++;
        }

        /* Storing the current cap information */
        for (i=0; i<n; i++)
            rr[0][i] = xx[i];

        /* Wait for non-blocking calls to complete */

        MPI_Waitall(idb, request, status);

        /* Storing the received cap information
         * XXX: this part assumes:
         *      1) E->sphere.caps_per_proc==1
         *      2) E->mesh.nsd==3
         */
        for (kk=0; kk<=num_ngb; kk++) {
            n = 0;
            for (i=1; i<=ncorners; i++) {
                theta = rr[kk][n++];
                phi = rr[kk][n++];
                rad = E->sphere.ro;

                sphere_to_cart(E, theta, phi, rad, &x, &y, &z);

                E->trace.xcap[kk][i] = x;
                E->trace.ycap[kk][i] = y;
                E->trace.zcap[kk][i] = z;
                E->trace.theta_cap[kk][i] = theta;
                E->trace.phi_cap[kk][i] = phi;
                E->trace.rad_cap[kk][i] = rad;
                E->trace.cos_theta[kk][i] = cos(theta);
                E->trace.sin_theta[kk][i] = sin(theta);
                E->trace.cos_phi[kk][i] = cos(phi);
                E->trace.sin_phi[kk][i] = sin(phi);
            }
        } /* end kk, number of neighbors */
}


/**************** INITIALIZE TRACER ARRAYS ************************************/
/*                                                                            */
/* This function allocates memories to tracer arrays.                         */

void allocate_tracer_arrays(struct All_variables *E, int number_of_tracers)
{

    int kk;

    /* max_ntracers is physical size of tracer array */
    /* (initially make it 25% larger than required */

    E->trace.max_ntracers[CPPR]=number_of_tracers+number_of_tracers/4;
    E->trace.ntracers[CPPR]=0;

    /* make tracer arrays */

    if ((E->trace.ielement[CPPR]=(int *) malloc(E->trace.max_ntracers[CPPR]*sizeof(int)))==NULL) {
        fprintf(E->trace.fpt,"ERROR(make tracer array)-no memory 1a\n");
        fflush(E->trace.fpt);
        exit(10);
    }
    for (kk=1;kk<E->trace.max_ntracers[CPPR];kk++)
        E->trace.ielement[CPPR][kk]=-99;


    for (kk=0;kk<E->trace.number_of_basic_quantities;kk++) {
        if ((E->trace.basicq[CPPR][kk]=(double *)malloc(E->trace.max_ntracers[CPPR]*sizeof(double)))==NULL) {
            fprintf(E->trace.fpt,"ERROR(initialize tracer arrays)-no memory 1b.%d\n",kk);
            fflush(E->trace.fpt);
            exit(10);
        }
    }

    for (kk=0;kk<E->trace.number_of_extra_quantities;kk++) {
        if ((E->trace.extraq[CPPR][kk]=(double *)malloc(E->trace.max_ntracers[CPPR]*sizeof(double)))==NULL) {
            fprintf(E->trace.fpt,"ERROR(initialize tracer arrays)-no memory 1c.%d\n",kk);
            fflush(E->trace.fpt);
            exit(10);
        }
    }

    if (E->trace.nflavors > 0) {
        E->trace.ntracer_flavor[CPPR]=(int **)malloc(E->trace.nflavors*sizeof(int*));
        for (kk=0;kk<E->trace.nflavors;kk++) {
            if ((E->trace.ntracer_flavor[CPPR][kk]=(int *)malloc((E->lmesh.nel+1)*sizeof(int)))==NULL) {
                fprintf(E->trace.fpt,"ERROR(initialize tracer arrays)-no memory 1c.%d\n",kk);
                fflush(E->trace.fpt);
                exit(10);
            }
        }
    }


    fprintf(E->trace.fpt,"Physical size of tracer arrays (max_ntracers): %d\n",
            E->trace.max_ntracers[CPPR]);
    fflush(E->trace.fpt);

    return;
}



/****** EXPAND TRACER ARRAYS *****************************************/

void expand_tracer_arrays(struct All_variables *E, int j)
{

    int inewsize;
    int kk;
    int icushion;

    /* expand basicq and ielement by 20% */

    icushion=100;

    inewsize=E->trace.max_ntracers[CPPR]+E->trace.max_ntracers[CPPR]/5+icushion;

    if ((E->trace.ielement[CPPR]=(int *)realloc(E->trace.ielement[CPPR],inewsize*sizeof(int)))==NULL) {
        fprintf(E->trace.fpt,"ERROR(expand tracer arrays )-no memory (ielement)\n");
        fflush(E->trace.fpt);
        exit(10);
    }

    for (kk=0;kk<=((E->trace.number_of_basic_quantities)-1);kk++) {
        if ((E->trace.basicq[CPPR][kk]=(double *)realloc(E->trace.basicq[CPPR][kk],inewsize*sizeof(double)))==NULL) {
            fprintf(E->trace.fpt,"ERROR(expand tracer arrays )-no memory (%d)\n",kk);
            fflush(E->trace.fpt);
            exit(10);
        }
    }

    for (kk=0;kk<=((E->trace.number_of_extra_quantities)-1);kk++) {
        if ((E->trace.extraq[CPPR][kk]=(double *)realloc(E->trace.extraq[CPPR][kk],inewsize*sizeof(double)))==NULL) {
            fprintf(E->trace.fpt,"ERROR(expand tracer arrays )-no memory 78 (%d)\n",kk);
            fflush(E->trace.fpt);
            exit(10);
        }
    }


    fprintf(E->trace.fpt,"Expanding physical memory of ielement, basicq, and extraq to %d from %d\n",
            inewsize,E->trace.max_ntracers[CPPR]);

    E->trace.max_ntracers[CPPR]=inewsize;

    return;
}




/****** REDUCE  TRACER ARRAYS *****************************************/

static void reduce_tracer_arrays(struct All_variables *E)
{

    int inewsize;
    int kk;
    int iempty_space;
    int j;

    int icushion=100;

        /* if physical size is double tracer size, reduce it */

        iempty_space=(E->trace.max_ntracers[CPPR]-E->trace.ntracers[CPPR]);

        if (iempty_space>(E->trace.ntracers[CPPR]+icushion)) {


            inewsize=E->trace.ntracers[CPPR]+E->trace.ntracers[CPPR]/4+icushion;

            if (inewsize<1) {
                fprintf(E->trace.fpt,"Error(reduce tracer arrays)-something up (hdf3)\n");
                fflush(E->trace.fpt);
                exit(10);
            }


            if ((E->trace.ielement[CPPR]=(int *)realloc(E->trace.ielement[CPPR],inewsize*sizeof(int)))==NULL) {
                fprintf(E->trace.fpt,"ERROR(reduce tracer arrays )-no memory (ielement)\n");
                fflush(E->trace.fpt);
                exit(10);
            }


            for (kk=0;kk<=((E->trace.number_of_basic_quantities)-1);kk++) {
                if ((E->trace.basicq[CPPR][kk]=(double *)realloc(E->trace.basicq[CPPR][kk],inewsize*sizeof(double)))==NULL) {
                    fprintf(E->trace.fpt,"AKM(reduce tracer arrays )-no memory (%d)\n",kk);
                    fflush(E->trace.fpt);
                    exit(10);
                }
            }

            for (kk=0;kk<=((E->trace.number_of_extra_quantities)-1);kk++) {
                if ((E->trace.extraq[CPPR][kk]=(double *)realloc(E->trace.extraq[CPPR][kk],inewsize*sizeof(double)))==NULL) {
                    fprintf(E->trace.fpt,"AKM(reduce tracer arrays )-no memory 783 (%d)\n",kk);
                    fflush(E->trace.fpt);
                    exit(10);
                }
            }


            fprintf(E->trace.fpt,"Reducing physical memory of ielement, basicq, and extraq to %d from %d\n",
                    E->trace.max_ntracers[CPPR],inewsize);

            E->trace.max_ntracers[CPPR]=inewsize;

        } /* end if */
}


/********** PUT AWAY LATER ************************************/
/*                                             */
/* rlater has a similar structure to basicq     */
/* ilatersize is the physical memory and       */
/* ilater is the number of tracers             */

static void put_away_later(struct All_variables *E, int j, int it)
{
    int kk;
    void expand_later_array();


    /* The first tracer in initiates memory allocation. */
    /* Memory is freed after parallel communications    */

    if (E->trace.ilatersize[CPPR]==0) {

        E->trace.ilatersize[CPPR]=E->trace.max_ntracers[CPPR]/5;

        for (kk=0;kk<=((E->trace.number_of_tracer_quantities)-1);kk++) {
            if ((E->trace.rlater[CPPR][kk]=(double *)malloc(E->trace.ilatersize[CPPR]*sizeof(double)))==NULL) {
                fprintf(E->trace.fpt,"AKM(put_away_later)-no memory (%d)\n",kk);
                fflush(E->trace.fpt);
                exit(10);
            }
        }
    } /* end first particle initiating memory allocation */


    /* Put tracer in later array */

    E->trace.ilater[CPPR]++;

    if (E->trace.ilater[CPPR] >= (E->trace.ilatersize[CPPR]-5)) expand_later_array(E,j);

    /* stack basic and extra quantities together (basic first) */

    for (kk=0;kk<=((E->trace.number_of_basic_quantities)-1);kk++)
        E->trace.rlater[CPPR][kk][E->trace.ilater[CPPR]]=E->trace.basicq[CPPR][kk][it];

    for (kk=0;kk<=((E->trace.number_of_extra_quantities)-1);kk++)
        E->trace.rlater[CPPR][E->trace.number_of_basic_quantities+kk][E->trace.ilater[CPPR]]=E->trace.extraq[CPPR][kk][it];


    return;
}


/****** EXPAND LATER ARRAY *****************************************/

void expand_later_array(struct All_variables *E, int j)
{

    int inewsize;
    int kk;
    int icushion;

    /* expand rlater by 20% */

    icushion=100;

    inewsize=E->trace.ilatersize[CPPR]+E->trace.ilatersize[CPPR]/5+icushion;

    for (kk=0;kk<=((E->trace.number_of_tracer_quantities)-1);kk++) {
        if ((E->trace.rlater[CPPR][kk]=(double *)realloc(E->trace.rlater[CPPR][kk],inewsize*sizeof(double)))==NULL) {
            fprintf(E->trace.fpt,"AKM(expand later array )-no memory (%d)\n",kk);
            fflush(E->trace.fpt);
            exit(10);
        }
    }

    fprintf(E->trace.fpt,"Expanding physical memory of rlater to %d from %d\n",
            inewsize,E->trace.ilatersize[CPPR]);

    E->trace.ilatersize[CPPR]=inewsize;
}


/***** EJECT TRACER ************************************************/

static void eject_tracer(struct All_variables *E, int it)
{

    int ilast_tracer;
    int kk;

    ilast_tracer=E->trace.ntracers[CPPR];

    /* put last tracer in ejected tracer position */

    E->trace.ielement[CPPR][it]=E->trace.ielement[CPPR][ilast_tracer];

    for (kk=0;kk<=((E->trace.number_of_basic_quantities)-1);kk++)
        E->trace.basicq[CPPR][kk][it]=E->trace.basicq[CPPR][kk][ilast_tracer];

    for (kk=0;kk<=((E->trace.number_of_extra_quantities)-1);kk++)
        E->trace.extraq[CPPR][kk][it]=E->trace.extraq[CPPR][kk][ilast_tracer];

    E->trace.ntracers[CPPR]--;
}



/********** ICHECK PROCESSOR SHELL *************/
/* returns -99 if rad is below current shell  */
/* returns 0 if rad is above current shell    */
/* returns 1 if rad is within current shell   */
/*                                            */
/* Shell, here, refers to processor shell     */
/*                                            */
/* shell is defined as bottom boundary up to  */
/* and not including the top boundary unless  */
/* the shell in question is the top shell     */

int icheck_processor_shell(struct All_variables *E,
                           int j, double rad)
{

    const int noz = E->lmesh.noz;
    const int nprocz = E->parallel.nprocz;
    double top_r, bottom_r;

    if (nprocz==1) return 1;

    top_r = E->sx[CPPR][3][noz];
    bottom_r = E->sx[CPPR][3][1];

    /* First check bottom */

    if (rad<bottom_r) return -99;


    /* Check top */

    if (rad<top_r) return 1;

    /* top processor */

    if ( (rad<=top_r) && (E->parallel.me_loc[3]==nprocz-1) ) return 1;

    /* If here, means point is above processor */
    return 0;
}


/********* ICHECK THAT PROCESSOR SHELL ********/
/*                                            */
/* Checks whether a given radius is within    */
/* a given processors radial domain.          */
/* Returns 0 if not, 1 if so.                 */
/* The domain is defined as including the bottom */
/* radius, but excluding the top radius unless   */
/* we the processor domain is the one that       */
/* is at the surface (then both boundaries are   */
/* included).                                    */

int icheck_that_processor_shell(struct All_variables *E,
                                int j, int nprocessor, double rad)
{
    int icheck_processor_shell();
    int me = E->parallel.me;

    /* nprocessor is right on top of me */
    if (nprocessor == me+1) {
        if (icheck_processor_shell(E, CPPR, rad) == 0) return 1;
        else return 0;
    }

    /* nprocessor is right on bottom of me */
    if (nprocessor == me-1) {
        if (icheck_processor_shell(E, CPPR, rad) == -99) return 1;
        else return 0;
    }

    /* Shouldn't be here */
    fprintf(E->trace.fpt, "Should not be here\n");
    fprintf(E->trace.fpt, "Error(check_shell) nprocessor: %d, radius: %f\n",
            nprocessor, rad);
    fflush(E->trace.fpt);
    exit(10);

    return 0;
}


