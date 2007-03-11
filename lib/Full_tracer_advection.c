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
#include "element_definitions.h"
#include "global_defs.h"
#include "parsing.h"
#include "parallel_related.h"
#include "composition_related.h"

void accumulate_tracers_in_element(struct All_variables *E);

static void get_neighboring_caps(struct All_variables *E);
static void pdebug(struct All_variables *E, int i);

static void fix_radius(struct All_variables *E,
                       double *radius, double *theta, double *phi,
                       double *x, double *y, double *z);
static void fix_theta_phi(double *theta, double *phi);
static void fix_phi(double *phi);
static void predict_tracers(struct All_variables *E);
static void correct_tracers(struct All_variables *E);
static int isum_tracers(struct All_variables *E);
static void make_tracer_array(struct All_variables *E);



/******* FULL TRACER INPUT *********************/

void full_tracer_input(E)
     struct All_variables *E;

{
    int m = E->parallel.me;

    /* Initial condition, this option is ignored if E->control.restart is 1,
    *  ie. restarted from a previous run */
    /* tracer_ic_method=0 (random generated array) */
    /* tracer_ic_method=1 (all proc read the same file) */
    /* tracer_ic_method=2 (each proc reads its restart file) */
    if(E->control.restart)
        E->trace.ic_method = 2;
    else {
        input_int("tracer_ic_method",&(E->trace.ic_method),"0,0,nomax",m);

        if (E->trace.ic_method==0)
            input_int("tracers_per_element",&(E->trace.itperel),"10,0,nomax",m);
        else if (E->trace.ic_method==1)
            input_string("tracer_file",E->trace.tracer_file,"tracer.dat",m);
        else if (E->trace.ic_method==2) {
        }
        else {
            fprintf(stderr,"Sorry, tracer_ic_method only 0, 1 and 2 available\n");
            fflush(stderr);
            parallel_process_termination();
        }
    }


    /* How many types of tracers, must be >= 1 */
    input_int("tracer_types",&(E->trace.ntypes),"1,1,nomax",m);



    /* Advection Scheme */

    /* itracer_advection_scheme=1 (simple predictor corrector -uses only V(to)) */
    /* itracer_advection_scheme=2 (predictor-corrector - uses V(to) and V(to+dt)) */

    E->trace.itracer_advection_scheme=2;
    input_int("tracer_advection_scheme",&(E->trace.itracer_advection_scheme),
              "2,0,nomax",m);

    if (E->trace.itracer_advection_scheme==1)
        {
        }
    else if (E->trace.itracer_advection_scheme==2)
        {
        }
    else
        {
            fprintf(stderr,"Sorry, only advection scheme 1 and 2 available (%d)\n",E->trace.itracer_advection_scheme);
            fflush(stderr);
            parallel_process_termination();
        }


    /* Interpolation Scheme */
    /* itracer_interpolation_scheme=1 (gnometric projection) */
    /* itracer_interpolation_scheme=2 (simple average) */

    E->trace.itracer_interpolation_scheme=1;
    input_int("tracer_interpolation_scheme",&(E->trace.itracer_interpolation_scheme),
              "1,0,nomax",m);
    if (E->trace.itracer_interpolation_scheme<1 || E->trace.itracer_interpolation_scheme>2)
        {
            fprintf(stderr,"Sorry, only interpolation scheme 1 and 2 available\n");
            fflush(stderr);
            parallel_process_termination();
        }

    /* Regular grid parameters */
    /* (first fill uniform del[0] value) */
    /* (later, in make_regular_grid, will adjust and distribute to caps */

    E->trace.deltheta[0]=1.0;
    E->trace.delphi[0]=1.0;
    input_double("regular_grid_deltheta",&(E->trace.deltheta[0]),"1.0",m);
    input_double("regular_grid_delphi",&(E->trace.delphi[0]),"1.0",m);


    /* Analytical Test Function */

    E->trace.ianalytical_tracer_test=0;
    input_int("analytical_tracer_test",&(E->trace.ianalytical_tracer_test),
              "0,0,nomax",m);


    composition_input(E);
    return;
}

/***** FULL TRACER SETUP ************************/

void full_tracer_setup(E)
     struct All_variables *E;
{

    char output_file[200];
    int m;
    void write_trace_instructions();
    void viscosity_checks();
    void initialize_old_composition();
    void find_tracers();
    void make_regular_grid();
    void initialize_tracer_elements();
    void define_uv_space();
    void determine_shape_coefficients();
    void check_sum();
    void read_tracer_file();
    void analytical_test();
    void tracer_post_processing();
    void restart_tracers();

    /* Some error control */

    if (E->sphere.caps_per_proc>1) {
            fprintf(stderr,"This code does not work for multiple caps per processor!\n");
            parallel_process_termination();
    }


    m=E->parallel.me;

    /* some obscure initial parameters */
    /* This parameter specifies how close a tracer can get to the boundary */
    E->trace.box_cushion=0.00001;

    /* AKMA turn this back on after debugging */
    E->trace.itracer_warnings=1;

    /* Determine number of tracer quantities */

    /* advection_quantites - those needed for advection */
    // TODO: generalize it
    if (E->trace.itracer_advection_scheme==1) E->trace.number_of_basic_quantities=12;
    if (E->trace.itracer_advection_scheme==2) E->trace.number_of_basic_quantities=12;

    /* extra_quantities - used for composition, etc.    */
    /* (can be increased for additional science i.e. tracing chemistry */

    E->trace.number_of_extra_quantities = 0;
    if (E->composition.ichemical_buoyancy==1)
        E->trace.number_of_extra_quantities += 1;


    E->trace.number_of_tracer_quantities =
        E->trace.number_of_basic_quantities +
        E->trace.number_of_extra_quantities;


    /* Fixed positions in tracer array */
    /* Comp is always in extraq position 0  */
    /* Current coordinates are always kept in basicq positions 0-5 */
    /* Other positions may be used depending on advection scheme and/or science being done */

    /* open tracing output file */

    sprintf(output_file,"%s.tracer_log.%d",E->control.data_file,m);
    E->trace.fpt=fopen(output_file,"w");


    /* reset statistical counters */

    E->trace.istat_isend=0;
    E->trace.istat_iempty=0;
    E->trace.istat_elements_checked=0;
    E->trace.istat1=0;

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

    write_trace_instructions(E);

    get_neighboring_caps(E);

    if (E->trace.ic_method==0) {
        make_tracer_array(E);

        if (E->composition.ichemical_buoyancy==1)
            init_tracer_composition(E);
    }
    else if (E->trace.ic_method==1) {
        read_tracer_file(E);

        if (E->composition.ichemical_buoyancy==1)
            init_tracer_composition(E);
    }
    else if (E->trace.ic_method==2) restart_tracers(E);
    else
        {
            fprintf(E->trace.fpt,"Not ready for other inputs yet\n");
            fflush(E->trace.fpt);
            exit(10);
        }

    /* flush and wait for not real reason but it can't hurt */
    fflush(E->trace.fpt);
    parallel_process_sync(E);


    if (E->trace.itracer_interpolation_scheme==1) {
        define_uv_space(E);
        determine_shape_coefficients(E);
    }


    /* flush and wait for not real reason but it can't hurt */
    fflush(E->trace.fpt);
    parallel_process_sync(E);

    make_regular_grid(E);

    /* flush and wait for not real reason but it can't hurt */
    fflush(E->trace.fpt);
    parallel_process_sync(E);


    /* find elements */

    find_tracers(E);

    /* total number of tracers  */

    E->trace.ilast_tracer_count = isum_tracers(E);
    fprintf(E->trace.fpt, "Sum of Tracers: %d\n", E->trace.ilast_tracer_count);

    if (E->trace.ianalytical_tracer_test==1) {
        //TODO: walk into this code...
        analytical_test(E);
        parallel_process_termination();
    }

    composition_setup(E);
    tracer_post_processing(E);

    return;
}


/******* TRACING *************************************************************/
/*                                                                           */
/* This function is the primary tracing routine called from Citcom.c         */
/* In this code, unlike the original 3D cartesian code, force is filled      */
/* during Stokes solution. No need to call thermal_buoyancy() after tracing. */


void full_tracer_advection(E)
     struct All_variables *E;
{

    void check_sum();
    void tracer_post_processing();


    fprintf(E->trace.fpt,"STEP %d\n",E->monitor.solution_cycles);
    fflush(E->trace.fpt);

    /* advect tracers */

    predict_tracers(E);
    correct_tracers(E);

    check_sum(E);

    //TODO: move
    if (E->composition.ichemical_buoyancy==1) {
        fill_composition(E);
    }

    tracer_post_processing(E);

    return;
}



/********* TRACER POST PROCESSING ****************************************/

void tracer_post_processing(E)
     struct All_variables *E;

{

    char output_file[200];

    double convection_time,tracer_time;
    double trace_fraction,total_time;

    void get_bulk_composition();

    static int been_here=0;

    //TODO: fix this function
    //if (E->composition.ichemical_buoyancy==1) get_bulk_composition(E);


    fprintf(E->trace.fpt,"Number of times for all element search  %d\n",E->trace.istat1);
    if (been_here!=0)
        {
            fprintf(E->trace.fpt,"Number of tracers sent to other processors: %d\n",E->trace.istat_isend);
            fprintf(E->trace.fpt,"Number of times element columns are checked: %d \n",E->trace.istat_elements_checked);
            if (E->composition.ichemical_buoyancy==1)
                {
                    fprintf(E->trace.fpt,"Empty elements filled with old compositional values: %d (%f percent)\n",
                            E->trace.istat_iempty,(100.0*E->trace.istat_iempty/E->lmesh.nel));
                }
        }

/*     fprintf(E->trace.fpt,"Error fraction: %f  (composition: %f)\n",E->trace.error_fraction,E->trace.bulk_composition); */



    /* reset statistical counters */

    E->trace.istat_isend=0;
    E->trace.istat_iempty=0;
    E->trace.istat_elements_checked=0;
    E->trace.istat1=0;

    /* compositional and error fraction data files */
    //TODO: move
    if (E->parallel.me==0)
        {
            if (been_here==0)
                {
                    if (E->composition.ichemical_buoyancy==1)
                        {
                            sprintf(output_file,"%s.error_fraction.data",E->control.data_file);
                            E->trace.fp_error_fraction=fopen(output_file,"w");

                            sprintf(output_file,"%s.composition.data",E->control.data_file);
                            E->trace.fp_composition=fopen(output_file,"w");
                        }
                }

            if (E->composition.ichemical_buoyancy==1)
                {
                    //TODO: to be init'd
                    fprintf(E->trace.fp_error_fraction,"%e %e\n",E->monitor.elapsed_time,E->trace.error_fraction);
                    fprintf(E->trace.fp_composition,"%e %e\n",E->monitor.elapsed_time,E->trace.bulk_composition);

                    fflush(E->trace.fp_error_fraction);
                    fflush(E->trace.fp_composition);
                }

        }



    fflush(E->trace.fpt);

    if (been_here==0) been_here++;

    return;
}


/*********** PREDICT TRACERS **********************************************/
/*                                                                        */
/* This function predicts tracers performing an euler step                */
/*                                                                        */
/*                                                                        */
/* Note positions used in tracer array                                    */
/* [positions 0-5 are always fixed with current coordinates               */
/*  regardless of advection scheme].                                      */
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

    void get_velocity();
    void get_cartesian_velocity_field();
    void cart_to_sphere();
    void keep_in_sphere();
    void find_tracers();

    static int been_here=0;

    dt=E->advection.timestep;

    /* if advection scheme is 2, don't have to calculate cartesian velocity again */
    /* (already did after last stokes calculation, unless this is first step)     */

    if ((been_here==0) && (E->trace.itracer_advection_scheme==2))
        {
            get_cartesian_velocity_field(E);
            been_here++;
        }

    if (E->trace.itracer_advection_scheme==1) get_cartesian_velocity_field(E);


    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {

            numtracers=E->trace.ntracers[j];

            for (kk=1;kk<=numtracers;kk++)
                {

                    theta0=E->trace.basicq[j][0][kk];
                    phi0=E->trace.basicq[j][1][kk];
                    rad0=E->trace.basicq[j][2][kk];
                    x0=E->trace.basicq[j][3][kk];
                    y0=E->trace.basicq[j][4][kk];
                    z0=E->trace.basicq[j][5][kk];

                    nelem=E->trace.ielement[j][kk];
                    get_velocity(E,j,nelem,theta0,phi0,rad0,velocity_vector);

                    x_pred=x0+velocity_vector[1]*dt;
                    y_pred=y0+velocity_vector[2]*dt;
                    z_pred=z0+velocity_vector[3]*dt;


                    /* keep in box */

                    cart_to_sphere(E,x_pred,y_pred,z_pred,&theta_pred,&phi_pred,&rad_pred);
                    keep_in_sphere(E,&x_pred,&y_pred,&z_pred,&theta_pred,&phi_pred,&rad_pred);

                    /* Current Coordinates are always kept in positions 0-5. */

                    E->trace.basicq[j][0][kk]=theta_pred;
                    E->trace.basicq[j][1][kk]=phi_pred;
                    E->trace.basicq[j][2][kk]=rad_pred;
                    E->trace.basicq[j][3][kk]=x_pred;
                    E->trace.basicq[j][4][kk]=y_pred;
                    E->trace.basicq[j][5][kk]=z_pred;

                    /* Fill in original coords (positions 6-8) */

                    E->trace.basicq[j][6][kk]=x0;
                    E->trace.basicq[j][7][kk]=y0;
                    E->trace.basicq[j][8][kk]=z0;

                    /* Fill in original velocities (positions 9-11) */

                    E->trace.basicq[j][9][kk]=velocity_vector[1];  /* Vx */
                    E->trace.basicq[j][10][kk]=velocity_vector[2];  /* Vy */
                    E->trace.basicq[j][11][kk]=velocity_vector[3];  /* Vz */


                } /* end kk, predicting tracers */
        } /* end caps */

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
/*  regardless of advection scheme].                                      */
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

    void get_velocity();
    void get_cartesian_velocity_field();
    void cart_to_sphere();
    void keep_in_sphere();
    void find_tracers();


    dt=E->advection.timestep;

    if ((E->parallel.me==0) && (E->trace.itracer_advection_scheme==2) )
        {
            fprintf(stderr,"AA:Correcting Tracers\n");
            fflush(stderr);
        }


    /* If scheme==1, use same velocity (t=0)      */
    /* Else if scheme==2, use new velocity (t=dt) */

    if (E->trace.itracer_advection_scheme==2)
        {
            get_cartesian_velocity_field(E);
        }


    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            for (kk=1;kk<=E->trace.ntracers[j];kk++)
                {

                    theta_pred=E->trace.basicq[j][0][kk];
                    phi_pred=E->trace.basicq[j][1][kk];
                    rad_pred=E->trace.basicq[j][2][kk];
                    x_pred=E->trace.basicq[j][3][kk];
                    y_pred=E->trace.basicq[j][4][kk];
                    z_pred=E->trace.basicq[j][5][kk];

                    x0=E->trace.basicq[j][6][kk];
                    y0=E->trace.basicq[j][7][kk];
                    z0=E->trace.basicq[j][8][kk];

                    Vx0=E->trace.basicq[j][9][kk];
                    Vy0=E->trace.basicq[j][10][kk];
                    Vz0=E->trace.basicq[j][11][kk];

                    nelem=E->trace.ielement[j][kk];

                    get_velocity(E,j,nelem,theta_pred,phi_pred,rad_pred,velocity_vector);

                    Vx_pred=velocity_vector[1];
                    Vy_pred=velocity_vector[2];
                    Vz_pred=velocity_vector[3];

                    x_cor=x0 + dt * 0.5*(Vx0+Vx_pred);
                    y_cor=y0 + dt * 0.5*(Vy0+Vy_pred);
                    z_cor=z0 + dt * 0.5*(Vz0+Vz_pred);

                    cart_to_sphere(E,x_cor,y_cor,z_cor,&theta_cor,&phi_cor,&rad_cor);
                    keep_in_sphere(E,&x_cor,&y_cor,&z_cor,&theta_cor,&phi_cor,&rad_cor);

                    /* Fill in Current Positions (other positions are no longer important) */

                    E->trace.basicq[j][0][kk]=theta_cor;
                    E->trace.basicq[j][1][kk]=phi_cor;
                    E->trace.basicq[j][2][kk]=rad_cor;
                    E->trace.basicq[j][3][kk]=x_cor;
                    E->trace.basicq[j][4][kk]=y_cor;
                    E->trace.basicq[j][5][kk]=z_cor;

                } /* end kk, correcting tracers */
        } /* end caps */
    /* find new tracer elements and caps */

    find_tracers(E);

    return;
}

/******** GET VELOCITY ***************************************/

void get_velocity(E,j,nelem,theta,phi,rad,velocity_vector)
     struct All_variables *E;
     int j,nelem;
     double theta,phi,rad;
     double *velocity_vector;
{

    void gnomonic_interpolation();

    /* gnomonic projection */

    if (E->trace.itracer_interpolation_scheme==1)
        {
            gnomonic_interpolation(E,j,nelem,theta,phi,rad,velocity_vector);
        }
    else if (E->trace.itracer_interpolation_scheme==2)
        {
            fprintf(E->trace.fpt,"Error(get velocity)-not ready for simple average interpolation scheme\n");
            fflush(E->trace.fpt);
            exit(10);
        }
    else
        {
            fprintf(E->trace.fpt,"Error(get velocity)-not ready for other interpolation schemes\n");
            fflush(E->trace.fpt);
            exit(10);
        }

    return;
}

/********************** GNOMONIC INTERPOLATION *********************************/
/*                                                                             */
/* This function interpolates tracer velocity using gnominic interpolation.    */
/* Real theta,phi,rad space is transformed into u,v space. This transformation */
/* maps great circles into straight lines. Here, elements boundaries are       */
/* assumed to be great circle planes (not entirely true, it is actually only   */
/* the nodal arrangement that lies upon great circles).  Element boundaries    */
/* are then mapped into planes.  The element is then divided into 2 wedges     */
/* in which standard shape functions are used to interpolate velocity.         */
/* This transformation was found on the internet (refs were difficult to       */
/* to obtain). It was tested that nodal configuration is indeed transformed    */
/* into straight lines.                                                        */
/* The transformations require a reference point along each cap. Here, the     */
/* midpoint is used.                                                           */
/* Radial and azimuthal shape functions are decoupled. First find the shape    */
/* functions associated with the 2D surface plane, then apply radial shape     */
/* functions.                                                                  */
/*                                                                             */
/* Wedge information:                                                          */
/*                                                                             */
/*        Wedge 1                  Wedge 2                                     */
/*        _______                  _______                                     */
/*                                                                             */
/*    wedge_node  real_node      wedge_node  real_node                         */
/*    ----------  ---------      ----------  ---------                         */
/*                                                                             */
/*         1        1               1            1                             */
/*         2        2               2            3                             */
/*         3        3               3            4                             */
/*         4        5               4            5                             */
/*         5        6               5            7                             */
/*         6        7               6            8                             */


void gnomonic_interpolation(E,j,nelem,theta,phi,rad,velocity_vector)
     struct All_variables *E;
     int j,nelem;
     double theta,phi,rad;
     double *velocity_vector;
{

    int iwedge,inum;
    int kk;
    int ival;
    int itry;

    double u,v;
    double shape2d[4];
    double shaperad[3];
    double shape[7];
    double vx[7],vy[7],vz[7];
    double x,y,z;


    int maxlevel=E->mesh.levmax;

    const double eps=-1e-4;

    void get_radial_shape();
    void sphere_to_cart();
    void spherical_to_uv();
    void get_2dshape();
    int iget_element();
    int icheck_element();
    int icheck_column_neighbors();


    /* find u and v using spherical coordinates */

    spherical_to_uv(E,j,theta,phi,&u,&v);

    inum=0;
    itry=1;

 try_again:

    /* Check first wedge (1 of 2) */

    iwedge=1;

 next_wedge:

    /* determine shape functions of wedge */
    /* There are 3 shape functions for the triangular wedge */

    get_2dshape(E,j,nelem,u,v,iwedge,shape2d);

    /* if any shape functions are negative, goto next wedge */

    if (shape2d[1]<eps||shape2d[2]<eps||shape2d[3]<eps)
        {
            inum=inum+1;
            /* AKMA clean this up */
            if (inum>3)
                {
                    fprintf(E->trace.fpt,"ERROR(gnomonic_interpolation)-inum>3!\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }
            if (inum>1 && itry==1)
                {
                    fprintf(E->trace.fpt,"ERROR(gnomonic_interpolation)-inum>1\n");
                    fprintf(E->trace.fpt,"shape %f %f %f\n",shape2d[1],shape2d[2],shape2d[3]);
                    fprintf(E->trace.fpt,"u %f v %f element: %d \n",u,v, nelem);
                    fprintf(E->trace.fpt,"Element uv boundaries: \n");
                    for(kk=1;kk<=4;kk++)
                        fprintf(E->trace.fpt,"%d: U: %f V:%f\n",kk,E->trace.UV[j][1][E->ien[j][nelem].node[kk]],E->trace.UV[j][2][E->ien[j][nelem].node[kk]]);
                    fprintf(E->trace.fpt,"theta: %f phi: %f rad: %f\n",theta,phi,rad);
                    fprintf(E->trace.fpt,"Element theta-phi boundaries: \n");
                    for(kk=1;kk<=4;kk++)
                        fprintf(E->trace.fpt,"%d: Theta: %f Phi:%f\n",kk,E->sx[j][1][E->ien[j][nelem].node[kk]],E->sx[j][2][E->ien[j][nelem].node[kk]]);
                    sphere_to_cart(E,theta,phi,rad,&x,&y,&z);
                    ival=icheck_element(E,j,nelem,x,y,z,rad);
                    fprintf(E->trace.fpt,"ICHECK?: %d\n",ival);
                    ival=iget_element(E,j,-99,x,y,z,theta,phi,rad);
                    fprintf(E->trace.fpt,"New Element?: %d\n",ival);
                    ival=icheck_column_neighbors(E,j,nelem,x,y,z,rad);
                    fprintf(E->trace.fpt,"New Element (neighs)?: %d\n",ival);
                    nelem=ival;
                    ival=icheck_element(E,j,nelem,x,y,z,rad);
                    fprintf(E->trace.fpt,"ICHECK?: %d\n",ival);
                    itry++;
                    if (ival>0) goto try_again;
                    fprintf(E->trace.fpt,"NO LUCK\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }

            iwedge=2;
            goto next_wedge;
        }

    /* Determine radial shape functions */
    /* There are 2 shape functions radially */

    get_radial_shape(E,j,nelem,rad,shaperad);

    /* There are 6 nodes to the solid wedge.             */
    /* The 6 shape functions assocated with the 6 nodes  */
    /* are products of radial and wedge shape functions. */

    /* Sum of shape functions is 1                       */

    shape[1]=shaperad[1]*shape2d[1];
    shape[2]=shaperad[1]*shape2d[2];
    shape[3]=shaperad[1]*shape2d[3];
    shape[4]=shaperad[2]*shape2d[1];
    shape[5]=shaperad[2]*shape2d[2];
    shape[6]=shaperad[2]*shape2d[3];

    /* depending on wedge, set up velocity points */

    if (iwedge==1)
        {
            vx[1]=E->trace.V0_cart[j][1][E->IEN[maxlevel][j][nelem].node[1]];
            vx[1]=E->trace.V0_cart[j][1][E->IEN[maxlevel][j][nelem].node[1]];
            vx[2]=E->trace.V0_cart[j][1][E->IEN[maxlevel][j][nelem].node[2]];
            vx[3]=E->trace.V0_cart[j][1][E->IEN[maxlevel][j][nelem].node[3]];
            vx[4]=E->trace.V0_cart[j][1][E->IEN[maxlevel][j][nelem].node[5]];
            vx[5]=E->trace.V0_cart[j][1][E->IEN[maxlevel][j][nelem].node[6]];
            vx[6]=E->trace.V0_cart[j][1][E->IEN[maxlevel][j][nelem].node[7]];
            vy[1]=E->trace.V0_cart[j][2][E->IEN[maxlevel][j][nelem].node[1]];
            vy[2]=E->trace.V0_cart[j][2][E->IEN[maxlevel][j][nelem].node[2]];
            vy[3]=E->trace.V0_cart[j][2][E->IEN[maxlevel][j][nelem].node[3]];
            vy[4]=E->trace.V0_cart[j][2][E->IEN[maxlevel][j][nelem].node[5]];
            vy[5]=E->trace.V0_cart[j][2][E->IEN[maxlevel][j][nelem].node[6]];
            vy[6]=E->trace.V0_cart[j][2][E->IEN[maxlevel][j][nelem].node[7]];
            vz[1]=E->trace.V0_cart[j][3][E->IEN[maxlevel][j][nelem].node[1]];
            vz[2]=E->trace.V0_cart[j][3][E->IEN[maxlevel][j][nelem].node[2]];
            vz[3]=E->trace.V0_cart[j][3][E->IEN[maxlevel][j][nelem].node[3]];
            vz[4]=E->trace.V0_cart[j][3][E->IEN[maxlevel][j][nelem].node[5]];
            vz[5]=E->trace.V0_cart[j][3][E->IEN[maxlevel][j][nelem].node[6]];
            vz[6]=E->trace.V0_cart[j][3][E->IEN[maxlevel][j][nelem].node[7]];
        }
    if (iwedge==2)
        {
            vx[1]=E->trace.V0_cart[j][1][E->IEN[maxlevel][j][nelem].node[1]];
            vx[2]=E->trace.V0_cart[j][1][E->IEN[maxlevel][j][nelem].node[3]];
            vx[3]=E->trace.V0_cart[j][1][E->IEN[maxlevel][j][nelem].node[4]];
            vx[4]=E->trace.V0_cart[j][1][E->IEN[maxlevel][j][nelem].node[5]];
            vx[5]=E->trace.V0_cart[j][1][E->IEN[maxlevel][j][nelem].node[7]];
            vx[6]=E->trace.V0_cart[j][1][E->IEN[maxlevel][j][nelem].node[8]];
            vy[1]=E->trace.V0_cart[j][2][E->IEN[maxlevel][j][nelem].node[1]];
            vy[2]=E->trace.V0_cart[j][2][E->IEN[maxlevel][j][nelem].node[3]];
            vy[3]=E->trace.V0_cart[j][2][E->IEN[maxlevel][j][nelem].node[4]];
            vy[4]=E->trace.V0_cart[j][2][E->IEN[maxlevel][j][nelem].node[5]];
            vy[5]=E->trace.V0_cart[j][2][E->IEN[maxlevel][j][nelem].node[7]];
            vy[6]=E->trace.V0_cart[j][2][E->IEN[maxlevel][j][nelem].node[8]];
            vz[1]=E->trace.V0_cart[j][3][E->IEN[maxlevel][j][nelem].node[1]];
            vz[2]=E->trace.V0_cart[j][3][E->IEN[maxlevel][j][nelem].node[3]];
            vz[3]=E->trace.V0_cart[j][3][E->IEN[maxlevel][j][nelem].node[4]];
            vz[4]=E->trace.V0_cart[j][3][E->IEN[maxlevel][j][nelem].node[5]];
            vz[5]=E->trace.V0_cart[j][3][E->IEN[maxlevel][j][nelem].node[7]];
            vz[6]=E->trace.V0_cart[j][3][E->IEN[maxlevel][j][nelem].node[8]];
        }

    velocity_vector[1]=vx[1]*shape[1]+vx[2]*shape[2]+shape[3]*vx[3]+
        vx[4]*shape[4]+vx[5]*shape[5]+shape[6]*vx[6];
    velocity_vector[2]=vy[1]*shape[1]+vy[2]*shape[2]+shape[3]*vy[3]+
        vy[4]*shape[4]+vy[5]*shape[5]+shape[6]*vy[6];
    velocity_vector[3]=vz[1]*shape[1]+vz[2]*shape[2]+shape[3]*vz[3]+
        vz[4]*shape[4]+vz[5]*shape[5]+shape[6]*vz[6];



    return;
}

/***************************************************************/
/* GET 2DSHAPE                                                 */
/*                                                             */
/* This function determines shape functions at u,v             */
/* This method uses standard linear shape functions of         */
/* triangular elements. (See Cuvelier, Segal, and              */
/* van Steenhoven, 1986).                                      */


void get_2dshape(E,j,nelem,u,v,iwedge,shape2d)
     struct All_variables *E;
     int j,nelem,iwedge;
     double u,v;
     double * shape2d;

{

    double a0,a1,a2;

    /* shape function 1 */

    a0=E->trace.shape_coefs[j][iwedge][1][nelem];
    a1=E->trace.shape_coefs[j][iwedge][2][nelem];
    a2=E->trace.shape_coefs[j][iwedge][3][nelem];

    shape2d[1]=a0+a1*u+a2*v;

    /* shape function 2 */

    a0=E->trace.shape_coefs[j][iwedge][4][nelem];
    a1=E->trace.shape_coefs[j][iwedge][5][nelem];
    a2=E->trace.shape_coefs[j][iwedge][6][nelem];

    shape2d[2]=a0+a1*u+a2*v;

    /* shape function 3 */

    a0=E->trace.shape_coefs[j][iwedge][7][nelem];
    a1=E->trace.shape_coefs[j][iwedge][8][nelem];
    a2=E->trace.shape_coefs[j][iwedge][9][nelem];

    shape2d[3]=a0+a1*u+a2*v;

    return;
}

/***************************************************************/
/* GET RADIAL SHAPE                                            */
/*                                                             */
/* This function determines radial shape functions at rad      */

void get_radial_shape(E,j,nelem,rad,shaperad)
     struct All_variables *E;
     int j,nelem;
     double rad;
     double * shaperad;

{

    int node1,node5;
    double rad1,rad5,f1,f2,delrad;

    const double eps=1e-6;
    double top_bound=1.0+eps;
    double bottom_bound=0.0-eps;

    node1=E->IEN[E->mesh.levmax][j][nelem].node[1];
    node5=E->IEN[E->mesh.levmax][j][nelem].node[5];

    rad1=E->sx[j][3][node1];
    rad5=E->sx[j][3][node5];

    delrad=rad5-rad1;

    f1=(rad-rad1)/delrad;
    f2=(rad5-rad)/delrad;

    /* Save a small amount of computation here   */
    /* because f1+f2=1, shapes can be switched   */
    /*
      shaperad[1]=1.0-f1=1.0-(1.0-f2)=f2;
      shaperad[2]=1.0-f2=1.0-(10-f1)=f1;
    */

    shaperad[1]=f2;
    shaperad[2]=f1;

    /* Some error control */

    if (shaperad[1]>(top_bound)||shaperad[1]<(bottom_bound)||
        shaperad[2]>(top_bound)||shaperad[2]<(bottom_bound))
        {
            fprintf(E->trace.fpt,"ERROR(get_radial_shape)\n");
            fprintf(E->trace.fpt,"shaperad[1]: %f \n",shaperad[1]);
            fprintf(E->trace.fpt,"shaperad[2]: %f \n",shaperad[2]);
            fflush(E->trace.fpt);
            exit(10);
        }

    return;
}





/**************************************************************/
/* SPHERICAL TO UV                                               */
/*                                                            */
/* This function transforms theta and phi to new coords       */
/* u and v using gnomonic projection.                          */

void spherical_to_uv(E,j,theta,phi,u,v)
     struct All_variables *E;
     int j;
     double theta,phi;
     double *u;
     double *v;

{

    double theta_f;
    double phi_f;
    double cosc;
    double cos_theta_f,sin_theta_f;
    double cost,sint,cosp2,sinp2;

    /* theta_f and phi_f are the reference points at the midpoint of the cap */

    theta_f=E->trace.UV[j][1][0];
    phi_f=E->trace.UV[j][2][0];

    cos_theta_f=E->trace.cos_theta_f;
    sin_theta_f=E->trace.sin_theta_f;

    cost=cos(theta);
    /*
      sint=sin(theta);
    */
    sint=sqrt(1.0-cost*cost);

    cosp2=cos(phi-phi_f);
    sinp2=sin(phi-phi_f);

    cosc=cos_theta_f*cost+sin_theta_f*sint*cosp2;
    cosc=1.0/cosc;

    *u=sint*sinp2*cosc;
    *v=(sin_theta_f*cost-cos_theta_f*sint*cosp2)*cosc;

    return;
}


/**************** INITIALIZE TRACER ARRAYS ************************************/
/*                                                                            */
/* This function allocates memories to tracer arrays.                         */

void initialize_tracer_arrays(E,j,number_of_tracers)
     struct All_variables *E;
     int j, number_of_tracers;
{

    int kk;

    /* max_ntracers is physical size of tracer array */
    /* (initially make it 25% larger than required */

    E->trace.max_ntracers[j]=number_of_tracers+number_of_tracers/4;
    E->trace.ntracers[j]=0;

    /* make tracer arrays */

    if ((E->trace.ielement[j]=(int *) malloc(E->trace.max_ntracers[j]*sizeof(int)))==NULL) {
        fprintf(E->trace.fpt,"ERROR(make tracer array)-no memory 1a\n");
        fflush(E->trace.fpt);
        exit(10);
    }

    for (kk=0;kk<E->trace.number_of_basic_quantities;kk++) {
        if ((E->trace.basicq[j][kk]=(double *)malloc(E->trace.max_ntracers[j]*sizeof(double)))==NULL) {
            fprintf(E->trace.fpt,"ERROR(initialize tracer arrays)-no memory 1b.%d\n",kk);
            fflush(E->trace.fpt);
            exit(10);
        }
    }

    for (kk=0;kk<E->trace.number_of_extra_quantities;kk++) {
        if ((E->trace.extraq[j][kk]=(double *)malloc(E->trace.max_ntracers[j]*sizeof(double)))==NULL) {
            fprintf(E->trace.fpt,"ERROR(initialize tracer arrays)-no memory 1c.%d\n",kk);
            fflush(E->trace.fpt);
            exit(10);
        }
    }

    fprintf(E->trace.fpt,"Physical size of tracer arrays (max_ntracers): %d\n",
            E->trace.max_ntracers[j]);
    fflush(E->trace.fpt);

    return;
}



/************ FIND TRACERS *************************************/
/*                                                             */
/* This function finds tracer elements and moves tracers to    */
/* other processor domains if necessary.                       */
/* Array ielement is filled with elemental values.                */

void find_tracers(E)
     struct All_variables *E;

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

    int iget_element();
    void put_away_later();
    void eject_tracer();
    void reduce_tracer_arrays();
    void lost_souls();
    void sphere_to_cart();

    static int been_here=0;

    time_stat1=CPU_time0();


    if (been_here==0)
        {
            for (j=1;j<=E->sphere.caps_per_proc;j++)
                {
                    for (kk=1;kk<=E->trace.ntracers[j];kk++)
                        {
                            E->trace.ielement[j][kk]=-99;
                        }
                }
            been_here++;
        }



    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {


            /* initialize arrays and statistical counters */

            E->trace.ilater[j]=0;

            E->trace.istat1=0;
            for (kk=0;kk<=4;kk++)
                {
                    E->trace.istat_ichoice[j][kk]=0;
                }

            //TODO: use while-loop instead of for-loop
            /* important to index by it, not kk */

            it=0;
            num_tracers=E->trace.ntracers[j];

            for (kk=1;kk<=num_tracers;kk++)
                {

                    it++;

                    theta=E->trace.basicq[j][0][it];
                    phi=E->trace.basicq[j][1][it];
                    rad=E->trace.basicq[j][2][it];
                    x=E->trace.basicq[j][3][it];
                    y=E->trace.basicq[j][4][it];
                    z=E->trace.basicq[j][5][it];

                    iprevious_element=E->trace.ielement[j][it];

                    /* AKMA REMOVE */
                    /*
                      fprintf(E->trace.fpt,"BB. kk %d %d %d %f %f %f %f %f %f\n",kk,j,iprevious_element,x,y,z,theta,phi,rad);
                      fflush(E->trace.fpt);
                    */
                    iel=iget_element(E,j,iprevious_element,x,y,z,theta,phi,rad);

                    E->trace.ielement[j][it]=iel;

                    if (iel<0)
                        {
                            put_away_later(E,j,it);
                            eject_tracer(E,j,it);
                            it--;
                        }

                } /* end tracers */

        } /* end j */


    /* Now take care of tracers that exited cap */

    /* REMOVE */
    /*
      parallel_process_termination();
    */

    lost_souls(E);

    /* Free later arrays */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            if (E->trace.ilater[j]>0)
                {
                    for (kk=0;kk<=((E->trace.number_of_tracer_quantities)-1);kk++)
                        {
                            free(E->trace.rlater[j][kk]);
                        }
                }
        } /* end j */


    /* Adjust Array Sizes */

    reduce_tracer_arrays(E);

    time_stat2=CPU_time0();

    fprintf(E->trace.fpt,"AA: time for find tracers: %f\n", time_stat2-time_stat1);

    return;
}

/************** LOST SOULS ****************************************************/
/*                                                                            */
/* This function is used to transport tracers to proper processor domains.    */
/* (MPI parallel)                                                             */
/*  All of the tracers that were sent to rlater arrays are destined to another */
/*  cap and sent there. Then they are raised up or down for multiple z procs.  */
/*  isend[j][n]=number of tracers this processor cap is sending to cap n        */
/*  ireceive[j][n]=number of tracers this processor cap is receiving from cap n */


void lost_souls(E)
     struct All_variables *E;
{
    int ithiscap;
    int ithatcap=1;
    int isend[13][13];
    int ireceive[13][13];
    int isize[13];
    int kk,pp,j;
    int mm;
    int numtracers;
    int icheck=0;
    int isend_position;
    int ipos,ipos2,ipos3;
    int idb;
    int idestination_proc=0;
    int isource_proc;
    int isend_z[13][3];
    int ireceive_z[13][3];
    int isum[13];
    int irad;
    int ival;
    int ithat_processor;
    int ireceive_position;
    int ihorizontal_neighbor;
    int ivertical_neighbor;
    int ilast_receiver_position;
    int it;
    int irec[13];
    int irec_position;
    int iel;
    int num_tracers;
    int isize_send;
    int isize_receive;
    int itemp_size;
    int itracers_subject_to_vertical_transport[13];

    double x,y,z;
    double theta,phi,rad;
    double *send[13][13];
    double *receive[13][13];
    double *send_z[13][3];
    double *receive_z[13][3];
    double *REC[13];

    int iget_element();
    int icheck_cap();
    void expand_tracer_arrays();

    int number_of_caps=12;
    int lev=E->mesh.levmax;
    int num_ngb;

    /* Note, if for some reason, the number of neighbors exceeds */
    /* 50, which is unlikely, the MPI arrays must be increased.  */
    MPI_Status status[200];
    MPI_Request request[200];
    MPI_Status status1;
    MPI_Status status2;
    static int itag=1;


    parallel_process_sync(E);
    fprintf(E->trace.fpt, "Entering lost_souls()\n");


    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            E->trace.istat_isend=E->trace.ilater[j];
        }

    for (j=1;j<=E->sphere.caps_per_proc;j++) {
        for (kk=1; kk<=E->trace.istat_isend; kk++) {
            fprintf(E->trace.fpt, "tracer#=%d xx=(%g,%g,%g)\n", kk,
                    E->trace.rlater[j][0][kk],
                    E->trace.rlater[j][1][kk],
                    E->trace.rlater[j][2][kk]);
        }
        fflush(E->trace.fpt);
    }

    /* initialize isend and ireceive */
    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            /* # of neighbors in the horizontal plane */
            num_ngb = E->parallel.TNUM_PASS[lev][j];
            isize[j]=E->trace.ilater[j]*E->trace.number_of_tracer_quantities;
            for (kk=1;kk<=num_ngb;kk++) isend[j][kk]=0;
            for (kk=1;kk<=num_ngb;kk++) ireceive[j][kk]=0;
        }

    /* Allocate Maximum Memory to Send Arrays */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {

            ithiscap=0;

            itemp_size=max(isize[j],1);

            if ((send[j][ithiscap]=(double *)malloc(itemp_size*sizeof(double)))==NULL)
                {
                    fprintf(E->trace.fpt,"Error(lost souls)-no memory (u389)\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }

            num_ngb = E->parallel.TNUM_PASS[lev][j];
            for (kk=1;kk<=num_ngb;kk++)
                {
                    if ((send[j][kk]=(double *)malloc(itemp_size*sizeof(double)))==NULL)
                        {
                            fprintf(E->trace.fpt,"Error(lost souls)-no memory (u389)\n");
                            fflush(E->trace.fpt);
                            exit(10);
                        }
                }
        }


    /* For testing, remove this */
    /*
      for (j=1;j<=E->sphere.caps_per_proc;j++)
      {
      ithiscap=E->sphere.capid[j];
      for (kk=1;kk<=E->parallel.TNUM_PASS[lev][j];kk++)
      {
          ithatcap=E->parallel.PROCESSOR[lev][j].pass[kk]+1;
          fprintf(E->trace.fpt,"cap: %d proc %d TNUM: %d ithatcap: %d\n",
                  ithiscap,E->parallel.me,kk,ithatcap);

      }
      fflush(E->trace.fpt);
      }
    */


    /* Pre communication */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {

            /* transfer tracers from rlater to send */

            numtracers=E->trace.ilater[j];

            for (kk=1;kk<=numtracers;kk++)
                {
                    rad=E->trace.rlater[j][2][kk];
                    x=E->trace.rlater[j][3][kk];
                    y=E->trace.rlater[j][4][kk];
                    z=E->trace.rlater[j][5][kk];

                    /* first check same cap if nprocz>1 */

                    if (E->parallel.nprocz>1)
                        {
                            ithatcap=0;
                            icheck=icheck_cap(E,ithatcap,x,y,z,rad);
                            if (icheck==1) goto foundit;

                        }

                    /* check neighboring caps */

                    for (pp=1;pp<=E->parallel.TNUM_PASS[lev][j];pp++)
                        {
                            ithatcap=pp;
                            icheck=icheck_cap(E,ithatcap,x,y,z,rad);
                            if (icheck==1) goto foundit;
                        }


                    /* should not be here */
                    if (icheck!=1)
                        {
                            fprintf(E->trace.fpt,"Error(lost souls)-should not be here\n");
                            fprintf(E->trace.fpt,"x: %f y: %f z: %f rad: %f\n",x,y,z,rad);
                            icheck=icheck_cap(E,0,x,y,z,rad);
                            if (icheck==1) fprintf(E->trace.fpt," icheck here!\n");
                            else fprintf(E->trace.fpt,"icheck not here!\n");
                            fflush(E->trace.fpt);
                            exit(10);
                        }

                foundit:

                    isend[j][ithatcap]++;

                    /* assign tracer to send */

                    isend_position=(isend[j][ithatcap]-1)*E->trace.number_of_tracer_quantities;

                    for (pp=0;pp<=(E->trace.number_of_tracer_quantities-1);pp++)
                        {
                            ipos=isend_position+pp;
                            send[j][ithatcap][ipos]=E->trace.rlater[j][pp][kk];
                        }

                } /* end kk, assigning tracers */

        } /* end j */


    /* Send info to other processors regarding number of send tracers */

    /* idb is the request array index variable */
    /* Each send and receive has a request variable */

    idb=0;
    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {

            ithiscap=0;

            /* if tracer is in same cap (nprocz>1) */

            if (E->parallel.nprocz>1)
                {
                    ireceive[j][ithiscap]=isend[j][ithiscap];
                }

            for (kk=1;kk<=E->parallel.TNUM_PASS[lev][j];kk++)
                {
                    ithatcap=kk;

                    /* if neighbor cap is in another processor, send information via MPI */

                    idestination_proc=E->parallel.PROCESSOR[lev][j].pass[kk];

                    idb++;
                    MPI_Isend(&isend[j][ithatcap],1,MPI_INT,idestination_proc,
                              11,E->parallel.world,
                              &request[idb-1]);

                } /* end kk, number of neighbors */

        } /* end j, caps per proc */

    /* Receive tracer count info */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            for (kk=1;kk<=E->parallel.TNUM_PASS[lev][j];kk++)
                {
                    ithatcap=kk;

                    /* if neighbor cap is in another processor, receive information via MPI */

                    isource_proc=E->parallel.PROCESSOR[lev][j].pass[kk];

                    if (idestination_proc!=E->parallel.me)
                        {

                            idb++;
                            MPI_Irecv(&ireceive[j][ithatcap],1,MPI_INT,isource_proc,
                                      11,E->parallel.world,
                                      &request[idb-1]);

                        } /* end if */

                } /* end kk, number of neighbors */
        } /* end j, caps per proc */

    /* Wait for non-blocking calls to complete */

    MPI_Waitall(idb,request,status);

    /* Testing, should remove */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
      {
      for (kk=1;kk<=E->parallel.TNUM_PASS[lev][j];kk++)
      {
          isource_proc=E->parallel.PROCESSOR[lev][j].pass[kk];
          fprintf(E->trace.fpt,"j: %d send %d to cap %d\n",j,isend[j][kk],isource_proc);
          fprintf(E->trace.fpt,"j: %d rec  %d from cap %d\n",j,ireceive[j][kk],isource_proc);
      }
      }


    /* Allocate memory in receive arrays */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            num_ngb = E->parallel.TNUM_PASS[lev][j];
            for (ithatcap=1;ithatcap<=num_ngb;ithatcap++)
                {
                    isize[j]=ireceive[j][ithatcap]*E->trace.number_of_tracer_quantities;

                    itemp_size=max(1,isize[j]);

                    if ((receive[j][ithatcap]=(double *)malloc(itemp_size*sizeof(double)))==NULL)
                        {
                            fprintf(E->trace.fpt,"Error(lost souls)-no memory (c721)\n");
                            fflush(E->trace.fpt);
                            exit(10);
                        }
                }
        } /* end j */

    /* Now, send the tracers to proper caps */

    idb=0;
    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            ithiscap=0;

            /* same cap */

            if (E->parallel.nprocz>1)
                {

                    ithatcap=ithiscap;
                    isize[j]=isend[j][ithatcap]*E->trace.number_of_tracer_quantities;
                    for (mm=0;mm<=(isize[j]-1);mm++)
                        {
                            receive[j][ithatcap][mm]=send[j][ithatcap][mm];
                        }

                }

            /* neighbor caps */

            for (kk=1;kk<=E->parallel.TNUM_PASS[lev][j];kk++)
                {
                    ithatcap=kk;

                    idestination_proc=E->parallel.PROCESSOR[lev][j].pass[kk];

                    isize[j]=isend[j][ithatcap]*E->trace.number_of_tracer_quantities;

                    idb++;

                    MPI_Isend(send[j][ithatcap],isize[j],MPI_DOUBLE,idestination_proc,
                              11,E->parallel.world,
                              &request[idb-1]);

                } /* end kk, number of neighbors */

        } /* end j, caps per proc */


    /* Receive tracers */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {

            ithiscap=0;
            for (kk=1;kk<=E->parallel.TNUM_PASS[lev][j];kk++)
                {
                    ithatcap=kk;

                    isource_proc=E->parallel.PROCESSOR[lev][j].pass[kk];

                    idb++;

                    isize[j]=ireceive[j][ithatcap]*E->trace.number_of_tracer_quantities;

                    MPI_Irecv(receive[j][ithatcap],isize[j],MPI_DOUBLE,isource_proc,
                              11,E->parallel.world,
                              &request[idb-1]);

                } /* end kk, number of neighbors */

        } /* end j, caps per proc */

    /* Wait for non-blocking calls to complete */

    MPI_Waitall(idb,request,status);


    /* Put all received tracers in array REC[j] */
    /* This makes things more convenient.       */

    /* Sum up size of receive arrays (all tracers sent to this processor) */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            isum[j]=0;

            ithiscap=0;

            for (kk=1;kk<=E->parallel.TNUM_PASS[lev][j];kk++)
                {
                    ithatcap=kk;
                    isum[j]=isum[j]+ireceive[j][ithatcap];
                }
            if (E->parallel.nprocz>1) isum[j]=isum[j]+ireceive[j][ithiscap];

            itracers_subject_to_vertical_transport[j]=isum[j];
        }

    /* Allocate Memory for REC array */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            isize[j]=isum[j]*E->trace.number_of_tracer_quantities;
            isize[j]=max(isize[j],1);
            if ((REC[j]=(double *)malloc(isize[j]*sizeof(double)))==NULL)
                {
                    fprintf(E->trace.fpt,"Error(lost souls)-no memory (g323)\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }
            REC[j][0]=0.0;
        }

    /* Put Received tracers in REC */


    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {

            irec[j]=0;

            irec_position=0;

            ithiscap=0;

            /* horizontal neighbors */

            for (ihorizontal_neighbor=1;ihorizontal_neighbor<=E->parallel.TNUM_PASS[lev][j];ihorizontal_neighbor++)
                {

                    ithatcap=ihorizontal_neighbor;

                    for (pp=1;pp<=ireceive[j][ithatcap];pp++)
                        {
                            irec[j]++;
                            ipos=(pp-1)*E->trace.number_of_tracer_quantities;

                            for (mm=0;mm<=(E->trace.number_of_tracer_quantities-1);mm++)
                                {
                                    ipos2=ipos+mm;
                                    REC[j][irec_position]=receive[j][ithatcap][ipos2];

                                    irec_position++;

                                } /* end mm (cycling tracer quantities) */
                        } /* end pp (cycling tracers) */
                } /* end ihorizontal_neighbors (cycling neighbors) */

            /* for tracers in the same cap (nprocz>1) */

            if (E->parallel.nprocz>1)
                {
                    ithatcap=ithiscap;
                    for (pp=1;pp<=ireceive[j][ithatcap];pp++)
                        {
                            irec[j]++;
                            ipos=(pp-1)*E->trace.number_of_tracer_quantities;

                            for (mm=0;mm<=(E->trace.number_of_tracer_quantities-1);mm++)
                                {
                                    ipos2=ipos+mm;

                                    REC[j][irec_position]=receive[j][ithatcap][ipos2];

                                    irec_position++;

                                } /* end mm (cycling tracer quantities) */

                        } /* end pp (cycling tracers) */

                } /* endif nproc>1 */

        } /* end j (cycling caps) */

    /* Done filling REC */



    /* VERTICAL COMMUNICATION */

    /* Note: For generality, I assume that both multiple */
    /* caps per processor as well as multiple processors */
    /* in the radial direction. These are probably       */
    /* inconsistent parameter choices for the regular    */
    /* CitcomS code.                                     */

    if (E->parallel.nprocz>1)
        {

            /* Allocate memory for send_z */
            /* Make send_z the size of receive array (max size) */
            /* (No dynamic reallocation of send_z necessary)    */

            for (j=1;j<=E->sphere.caps_per_proc;j++)
                {
                    for (kk=1;kk<=E->parallel.TNUM_PASSz[lev];kk++)
                        {
                            isize[j]=itracers_subject_to_vertical_transport[j]*E->trace.number_of_tracer_quantities;
                            isize[j]=max(isize[j],1);

                            if ((send_z[j][kk]=(double *)malloc(isize[j]*sizeof(double)))==NULL)
                                {
                                    fprintf(E->trace.fpt,"Error(lost souls)-no memory (c721)\n");
                                    fflush(E->trace.fpt);
                                    exit(10);
                                }
                        }
                } /* end j */


            for (j=1;j<=E->sphere.caps_per_proc;j++)
                {

                    for (ivertical_neighbor=1;ivertical_neighbor<=E->parallel.TNUM_PASSz[lev];ivertical_neighbor++)
                        {

                            ithat_processor=E->parallel.PROCESSORz[lev].pass[ivertical_neighbor];

                            /* initialize isend_z and ireceive_z array */

                            isend_z[j][ivertical_neighbor]=0;
                            ireceive_z[j][ivertical_neighbor]=0;

                            /* sort through receive array and check radius */

                            it=0;
                            num_tracers=irec[j];
                            for (kk=1;kk<=num_tracers;kk++)
                                {

                                    it++;

                                    ireceive_position=((it-1)*E->trace.number_of_tracer_quantities);

                                    irad=ireceive_position+2;

                                    rad=REC[j][irad];

                                    ival=icheck_that_processor_shell(E,j,ithat_processor,rad);


                                    /* if tracer is in other shell, take out of receive array and give to send_z*/

                                    if (ival==1)
                                        {

                                            isend_z[j][ivertical_neighbor]++;

                                            isend_position=(isend_z[j][ivertical_neighbor]-1)*E->trace.number_of_tracer_quantities;

                                            ilast_receiver_position=(irec[j]-1)*E->trace.number_of_tracer_quantities;

                                            for (mm=0;mm<=(E->trace.number_of_tracer_quantities-1);mm++)
                                                {
                                                    ipos=ireceive_position+mm;
                                                    ipos2=isend_position+mm;

                                                    send_z[j][ivertical_neighbor][ipos2]=REC[j][ipos];


                                                    /* eject tracer info from REC array, and replace with last tracer in array */

                                                    ipos3=ilast_receiver_position+mm;
                                                    REC[j][ipos]=REC[j][ipos3];

                                                }

                                            it--;
                                            irec[j]--;

                                        } /* end if ival===1 */

                                    /* Otherwise, leave tracer */

                                } /* end kk (cycling through tracers) */

                        } /* end ivertical_neighbor */

                } /* end j */


            /* Send arrays are now filled.                         */
            /* Now send send information to vertical processor neighbor */

            for (j=1;j<=E->sphere.caps_per_proc;j++)
                {
                    for (ivertical_neighbor=1;ivertical_neighbor<=E->parallel.TNUM_PASSz[lev];ivertical_neighbor++)
                        {

                            MPI_Sendrecv(&isend_z[j][ivertical_neighbor],1,MPI_INT,
                                         E->parallel.PROCESSORz[lev].pass[ivertical_neighbor],itag,
                                         &ireceive_z[j][ivertical_neighbor],1,MPI_INT,
                                         E->parallel.PROCESSORz[lev].pass[ivertical_neighbor],
                                         itag,E->parallel.world,&status1);

                            /* for testing - remove */
                            /*
                              fprintf(E->trace.fpt,"PROC: %d IVN: %d (P: %d) SEND: %d REC: %d\n",
                              E->parallel.me,ivertical_neighbor,E->parallel.PROCESSORz[lev].pass[ivertical_neighbor],
                              isend_z[j][ivertical_neighbor],ireceive_z[j][ivertical_neighbor]);
                              fflush(E->trace.fpt);
                            */

                        } /* end ivertical_neighbor */

                } /* end j */


            /* Allocate memory to receive_z arrays */


            for (j=1;j<=E->sphere.caps_per_proc;j++)
                {
                    for (kk=1;kk<=E->parallel.TNUM_PASSz[lev];kk++)
                        {
                            isize[j]=ireceive_z[j][kk]*E->trace.number_of_tracer_quantities;
                            isize[j]=max(isize[j],1);

                            if ((receive_z[j][kk]=(double *)malloc(isize[j]*sizeof(double)))==NULL)
                                {
                                    fprintf(E->trace.fpt,"Error(lost souls)-no memory (t590)\n");
                                    fflush(E->trace.fpt);
                                    exit(10);
                                }
                        }
                } /* end j */

            /* Send Tracers */

            for (j=1;j<=E->sphere.caps_per_proc;j++)
                {
                    for (ivertical_neighbor=1;ivertical_neighbor<=E->parallel.TNUM_PASSz[lev];ivertical_neighbor++)
                        {
                            isize_send=isend_z[j][ivertical_neighbor]*E->trace.number_of_tracer_quantities;
                            isize_receive=ireceive_z[j][ivertical_neighbor]*E->trace.number_of_tracer_quantities;

                            MPI_Sendrecv(send_z[j][ivertical_neighbor],isize_send,
                                         MPI_DOUBLE,
                                         E->parallel.PROCESSORz[lev].pass[ivertical_neighbor],itag+1,
                                         receive_z[j][ivertical_neighbor],isize_receive,
                                         MPI_DOUBLE,
                                         E->parallel.PROCESSORz[lev].pass[ivertical_neighbor],
                                         itag+1,E->parallel.world,&status2);

                        }
                }

            /* Put tracers into REC array */

            /* First, reallocate memory to REC */

            for (j=1;j<=E->sphere.caps_per_proc;j++)
                {
                    isum[j]=0;
                    for (ivertical_neighbor=1;ivertical_neighbor<=E->parallel.TNUM_PASSz[lev];ivertical_neighbor++)
                        {
                            isum[j]=isum[j]+ireceive_z[j][ivertical_neighbor];
                        }

                    isum[j]=isum[j]+irec[j];

                    isize[j]=isum[j]*E->trace.number_of_tracer_quantities;

                    if (isize[j]>0)
                        {
                            if ((REC[j]=(double *)realloc(REC[j],isize[j]*sizeof(double)))==NULL)
                                {
                                    fprintf(E->trace.fpt,"Error(lost souls)-no memory (i981)\n");
                                    fprintf(E->trace.fpt,"isize: %d\n",isize[j]);
                                    fflush(E->trace.fpt);
                                    exit(10);
                                }
                        }
                }


            for (j=1;j<=E->sphere.caps_per_proc;j++)
                {
                    for (ivertical_neighbor=1;ivertical_neighbor<=E->parallel.TNUM_PASSz[lev];ivertical_neighbor++)
                        {

                            for (kk=1;kk<=ireceive_z[j][ivertical_neighbor];kk++)
                                {
                                    irec[j]++;

                                    irec_position=(irec[j]-1)*E->trace.number_of_tracer_quantities;
                                    ireceive_position=(kk-1)*E->trace.number_of_tracer_quantities;

                                    for (mm=0;mm<=(E->trace.number_of_tracer_quantities-1);mm++)
                                        {
                                            REC[j][irec_position+mm]=receive_z[j][ivertical_neighbor][ireceive_position+mm];
                                        }
                                }

                        }
                }

            /* Free Vertical Arrays */

            for (j=1;j<=E->sphere.caps_per_proc;j++)
                {
                    for (ivertical_neighbor=1;ivertical_neighbor<=E->parallel.TNUM_PASSz[lev];ivertical_neighbor++)
                        {
                            free(send_z[j][ivertical_neighbor]);
                            free(receive_z[j][ivertical_neighbor]);
                        }
                }

        } /* endif nprocz>1 */

    /* END OF VERTICAL TRANSPORT */

    /* Put away tracers */


    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            for (kk=1;kk<=irec[j];kk++)
                {
                    E->trace.ntracers[j]++;

                    if (E->trace.ntracers[j]>(E->trace.max_ntracers[j]-5)) expand_tracer_arrays(E,j);

                    ireceive_position=(kk-1)*E->trace.number_of_tracer_quantities;

                    for (mm=0;mm<=(E->trace.number_of_basic_quantities-1);mm++)
                        {
                            ipos=ireceive_position+mm;

                            E->trace.basicq[j][mm][E->trace.ntracers[j]]=REC[j][ipos];
                        }
                    for (mm=0;mm<=(E->trace.number_of_extra_quantities-1);mm++)
                        {
                            ipos=ireceive_position+E->trace.number_of_basic_quantities+mm;

                            E->trace.extraq[j][mm][E->trace.ntracers[j]]=REC[j][ipos];
                        }

                    theta=E->trace.basicq[j][0][E->trace.ntracers[j]];
                    phi=E->trace.basicq[j][1][E->trace.ntracers[j]];
                    rad=E->trace.basicq[j][2][E->trace.ntracers[j]];
                    x=E->trace.basicq[j][3][E->trace.ntracers[j]];
                    y=E->trace.basicq[j][4][E->trace.ntracers[j]];
                    z=E->trace.basicq[j][5][E->trace.ntracers[j]];


                    iel=iget_element(E,j,-99,x,y,z,theta,phi,rad);

                    if (iel<1)
                        {
                            fprintf(E->trace.fpt,"Error(lost souls) - element not here?\n");
                            fprintf(E->trace.fpt,"x,y,z-theta,phi,rad: %f %f %f - %f %f %f\n",x,y,z,theta,phi,rad);
                            fflush(E->trace.fpt);
                            exit(10);
                        }

                    E->trace.ielement[j][E->trace.ntracers[j]]=iel;

                }
        }

    fprintf(E->trace.fpt,"Freeing memory in lost_souls()\n");
    fflush(E->trace.fpt);
    parallel_process_sync(E);

    /* Free Arrays */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {

            ithiscap=0;

            free(REC[j]);

            free(send[j][ithiscap]);

            for (kk=1;kk<=E->parallel.TNUM_PASS[lev][j];kk++)
                {
                    ithatcap=kk;

                    free(send[j][ithatcap]);
                    free(receive[j][ithatcap]);

                }

        }
    fprintf(E->trace.fpt,"Leaving lost_souls()\n");
    fflush(E->trace.fpt);

    return;
}


/****** REDUCE  TRACER ARRAYS *****************************************/

void reduce_tracer_arrays(E)
     struct All_variables *E;

{

    int inewsize;
    int kk;
    int iempty_space;
    int j;

    int icushion=100;

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {


            /* if physical size is double tracer size, reduce it */

            iempty_space=(E->trace.max_ntracers[j]-E->trace.ntracers[j]);

            if (iempty_space>(E->trace.ntracers[j]+icushion))
                {


                    inewsize=E->trace.ntracers[j]+E->trace.ntracers[j]/4+icushion;

                    if (inewsize<1)
                        {
                            fprintf(E->trace.fpt,"Error(reduce tracer arrays)-something up (hdf3)\n");
                            fflush(E->trace.fpt);
                            exit(10);
                        }


                    if ((E->trace.ielement[j]=(int *)realloc(E->trace.ielement[j],inewsize*sizeof(int)))==NULL)
                        {
                            fprintf(E->trace.fpt,"ERROR(reduce tracer arrays )-no memory (ielement)\n");
                            fflush(E->trace.fpt);
                            exit(10);
                        }


                    for (kk=0;kk<=((E->trace.number_of_basic_quantities)-1);kk++)
                        {
                            if ((E->trace.basicq[j][kk]=(double *)realloc(E->trace.basicq[j][kk],inewsize*sizeof(double)))==NULL)
                                {
                                    fprintf(E->trace.fpt,"AKM(reduce tracer arrays )-no memory (%d)\n",kk);
                                    fflush(E->trace.fpt);
                                    exit(10);
                                }
                        }

                    for (kk=0;kk<=((E->trace.number_of_extra_quantities)-1);kk++)
                        {
                            if ((E->trace.extraq[j][kk]=(double *)realloc(E->trace.extraq[j][kk],inewsize*sizeof(double)))==NULL)
                                {
                                    fprintf(E->trace.fpt,"AKM(reduce tracer arrays )-no memory 783 (%d)\n",kk);
                                    fflush(E->trace.fpt);
                                    exit(10);
                                }
                        }


                    fprintf(E->trace.fpt,"Reducing physical memory of ielement, basicq, and extraq to %d from %d\n",
                            E->trace.max_ntracers[j],inewsize);

                    E->trace.max_ntracers[j]=inewsize;

                } /* end if */

        } /* end j */

    return;
}

/********** PUT AWAY LATER ************************************/
/*                                             */
/* rlater has a similar structure to basicq     */
/* ilatersize is the physical memory and       */
/* ilater is the number of tracers             */


void put_away_later(E,j,it)
     struct All_variables *E;
     int it,j;

{


    int kk;


    void expand_later_array();


    /* The first tracer in initiates memory allocation. */
    /* Memory is freed after parallel communications    */

    if (E->trace.ilater[j]==0)
        {

            E->trace.ilatersize[j]=E->trace.max_ntracers[j]/5;

            for (kk=0;kk<=((E->trace.number_of_tracer_quantities)-1);kk++)
                {
                    if ((E->trace.rlater[j][kk]=(double *)malloc(E->trace.ilatersize[j]*sizeof(double)))==NULL)
                        {
                            fprintf(E->trace.fpt,"AKM(put_away_later)-no memory (%d)\n",kk);
                            fflush(E->trace.fpt);
                            exit(10);
                        }
                }
        } /* end first particle initiating memory allocation */


    /* Put tracer in later array */

    E->trace.ilater[j]++;

    if (E->trace.ilater[j] >= (E->trace.ilatersize[j]-5)) expand_later_array(E,j);

    /* stack basic and extra quantities together (basic first) */

    for (kk=0;kk<=((E->trace.number_of_basic_quantities)-1);kk++)
        {
            E->trace.rlater[j][kk][E->trace.ilater[j]]=E->trace.basicq[j][kk][it];
        }
    for (kk=0;kk<=((E->trace.number_of_extra_quantities)-1);kk++)
        {
            E->trace.rlater[j][E->trace.number_of_basic_quantities+kk][E->trace.ilater[j]]=E->trace.extraq[j][kk][it];
        }

    return;
}

/****** EXPAND LATER ARRAY *****************************************/

void expand_later_array(E,j)
     struct All_variables *E;
     int j;
{

    int inewsize;
    int kk;
    int icushion;

    /* expand rlater by 20% */

    icushion=100;

    inewsize=E->trace.ilatersize[j]+E->trace.ilatersize[j]/5+icushion;

    for (kk=0;kk<=((E->trace.number_of_tracer_quantities)-1);kk++)
        {
            if ((E->trace.rlater[j][kk]=(double *)realloc(E->trace.rlater[j][kk],inewsize*sizeof(double)))==NULL)
                {
                    fprintf(E->trace.fpt,"AKM(expand later array )-no memory (%d)\n",kk);
                    fflush(E->trace.fpt);
                    exit(10);
                }
        }


    fprintf(E->trace.fpt,"Expanding physical memory of rlater to %d from %d\n",
            inewsize,E->trace.ilatersize[j]);

    E->trace.ilatersize[j]=inewsize;

    return;
}


/***** EJECT TRACER ************************************************/


void eject_tracer(E,j,it)
     struct All_variables *E;
     int it,j;

{

    int ilast_tracer;
    int kk;


    ilast_tracer=E->trace.ntracers[j];

    /* put last tracer in ejected tracer position */

    E->trace.ielement[j][it]=E->trace.ielement[j][ilast_tracer];

    for (kk=0;kk<=((E->trace.number_of_basic_quantities)-1);kk++)
        {
            E->trace.basicq[j][kk][it]=E->trace.basicq[j][kk][ilast_tracer];
        }
    for (kk=0;kk<=((E->trace.number_of_extra_quantities)-1);kk++)
        {
            E->trace.extraq[j][kk][it]=E->trace.extraq[j][kk][ilast_tracer];
        }


    E->trace.ntracers[j]--;

    return;
}



/*********** MAKE REGULAR GRID ********************************/
/*                                                            */
/* This function generates the finer regular grid which is    */
/* mapped to real elements                                    */

void make_regular_grid(E)
     struct All_variables *E;
{

    int j;
    int kk;
    int mm;
    int pp,node;
    int numtheta,numphi;
    int nodestheta,nodesphi;
    unsigned int numregel;
    unsigned int numregnodes;
    int idum1,idum2;
    int ifound_one;
    int ival;
    int ilast_el;
    int imap;
    int elz;
    int nelsurf;
    int iregnode[5];
    int ntheta,nphi;
    int ichoice;
    int icount;
    int itemp[5];
    int iregel;
    int istat_ichoice[13][5];
    int isum;

    double x,y,z;
    double theta,phi,rad;
    double deltheta;
    double delphi;
    double thetamax,thetamin;
    double phimax,phimin;
    double start_time;
    double theta_min,phi_min;
    double theta_max,phi_max;
    double half_diff;
    double expansion;

    double *tmin;
    double *tmax;
    double *fmin;
    double *fmax;

    int icheck_all_columns();
    int icheck_element_column();
    int icheck_column_neighbors();
    void sphere_to_cart();

    const double two_pi=2.0*M_PI;

    elz=E->lmesh.elz;

    nelsurf=E->lmesh.elx*E->lmesh.ely;

    //TODO: find the bounding box of the mesh, if the box is too close to
    // to core, set a flag (rotated_reggrid) to true and rotate the
    // bounding box to the equator. Generate the regular grid with the new
    // bounding box. The rotation should be a simple one, e.g.
    // (theta, phi) -> (??)
    // Whenever the regular grid is used, check the flat (rotated_reggrid),
    // if true, rotate the checkpoint as well.

    /* note, mesh is rotated along theta 22.5 degrees divided by elx. */
    /* We at least want that much expansion here! Otherwise, theta min */
    /* will not be valid near poles. We do a little more (x2) to be safe */
    /* Typically 1-2 degrees. Look in nodal_mesh.c for this.             */

    expansion=2.0*0.5*(M_PI/4.0)/(1.0*E->lmesh.elx);

    start_time=CPU_time0();

    if (E->parallel.me==0) fprintf(stderr,"Generating Regular Grid\n");
    fflush(stderr);


    /* for each cap, determine theta and phi bounds, watch out near poles  */

    numregnodes=0;
    for(j=1;j<=E->sphere.caps_per_proc;j++)
        {

            thetamax=0.0;
            thetamin=M_PI;

            phimax=two_pi;
            phimin=0.0;

            for (kk=1;kk<=E->lmesh.nno;kk=kk+E->lmesh.noz)
                {

                    theta=E->sx[j][1][kk];
                    phi=E->sx[j][2][kk];

                    thetamax=max(thetamax,theta);
                    thetamin=min(thetamin,theta);

                }

            /* expand range slightly (should take care of poles)  */

            thetamax=thetamax+expansion;
            thetamax=min(thetamax,M_PI);

            thetamin=thetamin-expansion;
            thetamin=max(thetamin,0.0);

            /* Convert input data from degrees to radians  */

            deltheta=E->trace.deltheta[0]*M_PI/180.0;
            delphi=E->trace.delphi[0]*M_PI/180.0;


            /* Adjust deltheta and delphi to fit a uniform number of regular elements */

            numtheta=fabs(thetamax-thetamin)/deltheta;
            numphi=fabs(phimax-phimin)/delphi;
            nodestheta=numtheta+1;
            nodesphi=numphi+1;
            numregel=numtheta*numphi;
            numregnodes=nodestheta*nodesphi;

            if ((numtheta==0)||(numphi==0))
                {
                    fprintf(E->trace.fpt,"Error(make_regular_grid): numtheta: %d numphi: %d\n",numtheta,numphi);
                    fflush(E->trace.fpt);
                    exit(10);
                }

            deltheta=fabs(thetamax-thetamin)/(1.0*numtheta);
            delphi=fabs(phimax-phimin)/(1.0*numphi);

            /* fill global variables */

            E->trace.deltheta[j]=deltheta;
            E->trace.delphi[j]=delphi;
            E->trace.numtheta[j]=numtheta;
            E->trace.numphi[j]=numphi;
            E->trace.thetamax[j]=thetamax;
            E->trace.thetamin[j]=thetamin;
            E->trace.phimax[j]=phimax;
            E->trace.phimin[j]=phimin;
            E->trace.numregel[j]=numregel;
            E->trace.numregnodes[j]=numregnodes;

            if ( ((1.0*numregel)/(1.0*E->lmesh.elx*E->lmesh.ely)) < 0.5 )
                {
                    fprintf(E->trace.fpt,"\n ! WARNING: regular/real ratio low: %f ! \n",
                            ((1.0*numregel)/(1.0*E->lmesh.nel)) );
                    fprintf(E->trace.fpt," Should reduce size of regular mesh\n");
                    fprintf(stderr,"! WARNING: regular/real ratio low: %f ! \n",
                            ((1.0*numregel)/(1.0*E->lmesh.nel)) );
                    fprintf(stderr," Should reduce size of regular mesh\n");
                    fflush(E->trace.fpt);
                    fflush(stderr);
                    if (E->trace.itracer_warnings==1) exit(10);
                }

            /* print some output */

            fprintf(E->trace.fpt,"\nRegular grid:\n");
            fprintf(E->trace.fpt,"Theta min: %f max: %f \n",thetamin,thetamax);
            fprintf(E->trace.fpt,"Phi min: %f max: %f \n",phimin,phimax);
            fprintf(E->trace.fpt,"Adjusted deltheta: %f delphi: %f\n",deltheta,delphi);
            fprintf(E->trace.fpt,"(numtheta: %d  numphi: %d)\n",numtheta,numphi);
            fprintf(E->trace.fpt,"Number of regular elements: %d  (nodes: %d)\n",numregel,numregnodes);

            fprintf(E->trace.fpt,"regular/real ratio: %f\n",((1.0*numregel)/(1.0*E->lmesh.elx*E->lmesh.ely)));
            fflush(E->trace.fpt);

            /* Allocate memory for regnodetoel */
            /* Regtoel is an integer array which represents nodes on    */
            /* the regular mesh. Each node on the regular mesh contains */
            /* the real element value if one exists (-99 otherwise)     */



            if ((E->trace.regnodetoel[j]=(int *)malloc((numregnodes+1)*sizeof(int)))==NULL)
                {
                    fprintf(E->trace.fpt,"ERROR(make regular) -no memory - uh3ud\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }


            /* Initialize regnodetoel - reg elements not used =-99 */

            for (kk=1;kk<=numregnodes;kk++)
                {
                    E->trace.regnodetoel[j][kk]=-99;
                }

            /* Begin Mapping (only need to use surface elements) */

            parallel_process_sync(E);
            if (E->parallel.me==0) fprintf(stderr,"Beginning Mapping\n");
            fflush(stderr);

            /* Generate temporary arrays of max and min values for each surface element */


            if ((tmin=(double *)malloc((nelsurf+1)*sizeof(double)))==NULL)
                {
                    fprintf(E->trace.fpt,"ERROR(make regular) -no memory - 7t1a\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }
            if ((tmax=(double *)malloc((nelsurf+1)*sizeof(double)))==NULL)
                {
                    fprintf(E->trace.fpt,"ERROR(make regular) -no memory - 7t1a\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }
            if ((fmin=(double *)malloc((nelsurf+1)*sizeof(double)))==NULL)
                {
                    fprintf(E->trace.fpt,"ERROR(make regular) -no memory - 7t1a\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }
            if ((fmax=(double *)malloc((nelsurf+1)*sizeof(double)))==NULL)
                {
                    fprintf(E->trace.fpt,"ERROR(make regular) -no memory - 7t1a\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }

            for (mm=elz;mm<=E->lmesh.nel;mm=mm+elz)
                {

                    kk=mm/elz;

                    theta_min=M_PI;
                    theta_max=0.0;
                    phi_min=two_pi;
                    phi_max=0.0;
                    for (pp=1;pp<=4;pp++)
                        {
                            node=E->ien[j][mm].node[pp];
                            theta=E->sx[j][1][node];
                            phi=E->sx[j][2][node];

                            theta_min=min(theta_min,theta);
                            theta_max=max(theta_max,theta);
                            phi_min=min(phi_min,phi);
                            phi_max=max(phi_max,phi);
                        }

                    /* add half difference to phi and expansion to theta to be safe */

                    theta_max=theta_max+expansion;
                    theta_min=theta_min-expansion;

                    theta_max=min(M_PI,theta_max);
                    theta_min=max(0.0,theta_min);

                    half_diff=0.5*(phi_max-phi_min);
                    phi_max=phi_max+half_diff;
                    phi_min=phi_min-half_diff;

                    fix_phi(&phi_max);
                    fix_phi(&phi_min);

                    if (phi_min>phi_max)
                        {
                            phi_min=0.0;
                            phi_max=two_pi;
                        }

                    tmin[kk]=theta_min;
                    tmax[kk]=theta_max;
                    fmin[kk]=phi_min;
                    fmax[kk]=phi_max;
                }

            /* end looking through elements */

            ifound_one=0;

            rad=E->sphere.ro;

            imap=0;

            for (kk=1;kk<=numregnodes;kk++)
                {

                    E->trace.regnodetoel[j][kk]=-99;

                    /* find theta and phi for a given regular node */

                    idum1=(kk-1)/(numtheta+1);
                    idum2=kk-1-idum1*(numtheta+1);

                    theta=thetamin+(1.0*idum2*deltheta);
                    phi=phimin+(1.0*idum1*delphi);

                    sphere_to_cart(E,theta,phi,rad,&x,&y,&z);


                    ilast_el=1;

                    /* if previous element not found yet, check all surface elements */

                    /*
                      if (ifound_one==0)
                      {
                      for (mm=elz;mm<=E->lmesh.nel;mm=mm+elz)
                      {
                      pp=mm/elz;
                      if ( (theta>=tmin[pp]) && (theta<=tmax[pp]) && (phi>=fmin[pp]) && (phi<=fmax[pp]) )
                      {
                      ival=icheck_element_column(E,j,mm,x,y,z,rad);
                      if (ival>0)
                      {
                      ilast_el=mm;
                      ifound_one++;
                      E->trace.regnodetoel[j][kk]=mm;
                      goto foundit;
                      }
                      }
                      }
                      goto foundit;
                      }
                    */

                    /* first check previous element */

                    ival=icheck_element_column(E,j,ilast_el,x,y,z,rad);
                    if (ival>0)
                        {
                            E->trace.regnodetoel[j][kk]=ilast_el;
                            goto foundit;
                        }

                    /* check neighbors */

                    ival=icheck_column_neighbors(E,j,ilast_el,x,y,z,rad);
                    if (ival>0)
                        {
                            E->trace.regnodetoel[j][kk]=ival;
                            ilast_el=ival;
                            goto foundit;
                        }

                    /* check all */

                    for (mm=elz;mm<=E->lmesh.nel;mm=mm+elz)
                        {
                            pp=mm/elz;
                            if ( (theta>=tmin[pp]) && (theta<=tmax[pp]) && (phi>=fmin[pp]) && (phi<=fmax[pp]) )
                                {
                                    ival=icheck_element_column(E,j,mm,x,y,z,rad);
                                    if (ival>0)
                                        {
                                            ilast_el=mm;
                                            E->trace.regnodetoel[j][kk]=mm;
                                            goto foundit;
                                        }
                                }
                        }

                foundit:

                    if (E->trace.regnodetoel[j][kk]>0) imap++;

                } /* end all regular nodes (kk) */

            fprintf(E->trace.fpt,"percentage mapped: %f\n", (1.0*imap)/(1.0*numregnodes)*100.0);
            fflush(E->trace.fpt);

            /* free temporary arrays */

            free(tmin);
            free(tmax);
            free(fmin);
            free(fmax);

        } /* end j */


    /* some error control */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            for (kk=1;kk<=numregnodes;kk++)
                {

                    if (E->trace.regnodetoel[j][kk]!=-99)
                        {
                            if ( (E->trace.regnodetoel[j][kk]<1)||(E->trace.regnodetoel[j][kk]>E->lmesh.nel) )
                                {
                                    fprintf(stderr,"Error(make_regular_grid)-invalid element: %d\n",E->trace.regnodetoel[j][kk]);
                                    fprintf(E->trace.fpt,"Error(make_regular_grid)-invalid element: %d\n",E->trace.regnodetoel[j][kk]);
                                    fflush(E->trace.fpt);
                                    fflush(stderr);
                                    exit(10);
                                }
                        }
                }
        }


    /* Now put regnodetoel information into regtoel */


    if (E->parallel.me==0) fprintf(stderr,"Beginning Regtoel submapping \n");
    fflush(stderr);

    /* AKMA decided it would be more efficient to have reg element choice array */
    /* rather than reg node array as used before          */


    for(j=1;j<=E->sphere.caps_per_proc;j++)
        {

            /* initialize statistical counter */

            for (pp=0;pp<=4;pp++) istat_ichoice[j][pp]=0;

            /* Allocate memory for regtoel */
            /* Regtoel consists of 4 positions for each regular element */
            /* Position[0] lists the number of element choices (later   */
            /* referred to as ichoice), followed                        */
            /* by the possible element choices.                          */
            /* ex.) A regular element has 4 nodes. Each node resides in  */
            /* a real element. The number of real elements a regular     */
            /* element touches (one of its nodes are in) is ichoice.     */
            /* Special ichoice notes:                                    */
            /*    ichoice=-1   all regular element nodes = -99 (no elements) */
            /*    ichoice=0    all 4 corners within one element              */
            /*    ichoice=1     one element choice (diff from ichoice 0 in    */
            /*                  that perhaps one reg node is in an element    */
            /*                  and the rest are not (-99).                   */
            /*    ichoice>1     Multiple elements to check                    */

            numregel= E->trace.numregel[j];

            for (pp=0;pp<=4;pp++)
                {
                    if ((E->trace.regtoel[j][pp]=(int *)malloc((numregel+1)*sizeof(int)))==NULL)
                        {
                            fprintf(E->trace.fpt,"ERROR(make regular)-no memory 98d (%d %d %d)\n",pp,numregel,j);
                            fflush(E->trace.fpt);
                            exit(10);
                        }
                }

            numtheta=E->trace.numtheta[j];
            numphi=E->trace.numphi[j];

            for (nphi=1;nphi<=numphi;nphi++)
                {
                    for (ntheta=1;ntheta<=numtheta;ntheta++)
                        {

                            iregel=ntheta+(nphi-1)*numtheta;

                            /* initialize regtoel (not necessary really) */

                            for (pp=0;pp<=4;pp++) E->trace.regtoel[j][pp][iregel]=-33;

                            if ( (iregel>numregel)||(iregel<1) )
                                {
                                    fprintf(E->trace.fpt,"ERROR(make_regular_grid)-weird iregel: %d (max: %d)\n",iregel,numregel);
                                    fflush(E->trace.fpt);
                                    exit(10);
                                }

                            iregnode[1]=iregel+(nphi-1);
                            iregnode[2]=iregel+nphi;
                            iregnode[3]=iregel+nphi+E->trace.numtheta[j]+1;
                            iregnode[4]=iregel+nphi+E->trace.numtheta[j];

                            for (kk=1;kk<=4;kk++)
                                {
                                    if ((iregnode[kk]<1)||(iregnode[kk]>numregnodes))
                                        {
                                            fprintf(E->trace.fpt,"ERROR(make regular)-bad regnode %d\n",iregnode[kk]);
                                            fflush(E->trace.fpt);
                                            exit(10);
                                        }
                                    if (E->trace.regnodetoel[j][iregnode[kk]]>E->lmesh.nel)
                                        {
                                            fprintf(E->trace.fpt,"AABB HERE %d %d %d %d\n",iregel,iregnode[kk],kk,E->trace.regnodetoel[j][iregnode[kk]]);
                                            fflush(E->trace.fpt);
                                        }
                                }


                            /* find number of choices */

                            ichoice=0;
                            icount=0;

                            for (kk=1;kk<=4;kk++)
                                {

                                    if (E->trace.regnodetoel[j][iregnode[kk]]<=0) goto next_corner;

                                    icount++;
                                    for (pp=1;pp<=(kk-1);pp++)
                                        {
                                            if (E->trace.regnodetoel[j][iregnode[kk]]==E->trace.regnodetoel[j][iregnode[pp]]) goto next_corner;
                                        }
                                    ichoice++;
                                    itemp[ichoice]=E->trace.regnodetoel[j][iregnode[kk]];

                                    if ((ichoice<0) || (ichoice>4) )
                                        {
                                            fprintf(E->trace.fpt,"ERROR(make regular) - weird ichoice %d \n",ichoice);
                                            fflush(E->trace.fpt);
                                            exit(10);
                                        }
                                    if ((itemp[ichoice]<0) || (itemp[ichoice]>E->lmesh.nel) )
                                        {
                                            fprintf(E->trace.fpt,"ERROR(make regular) - weird element choice %d %d\n",itemp[ichoice],ichoice);
                                            fflush(E->trace.fpt);
                                            exit(10);
                                        }

                                next_corner:
                                    ;
                                } /* end kk */

                            istat_ichoice[j][ichoice]++;

                            if ((ichoice<0) || (ichoice>4))
                                {
                                    fprintf(E->trace.fpt,"ERROR(make_regular)-wierd ichoice %d\n",ichoice);
                                    fflush(E->trace.fpt);
                                    exit(10);
                                }

                            if (ichoice==0)
                                {
                                    E->trace.regtoel[j][0][iregel]=-1;
                                    /*
                                      fprintf(E->trace.fpt,"HH1: (%p) iregel: %d ichoice: %d value: %d %d\n",&E->trace.regtoel[j][1][iregel],iregel,ichoice,E->trace.regtoel[j][0][iregel],E->trace.regtoel[j][1][iregel]);
                                    */
                                }
                            else if ( (ichoice==1) && (icount==4) )
                                {
                                    E->trace.regtoel[j][0][iregel]=0;
                                    E->trace.regtoel[j][1][iregel]=itemp[1];

                                    /*
                                      fprintf(E->trace.fpt,"HH2: (%p) iregel: %d ichoice: %d value: %d %d\n",&E->trace.regtoel[j][1][iregel],iregel,ichoice,E->trace.regtoel[j][0][iregel],E->trace.regtoel[j][1][iregel]);
                                    */

                                    if (itemp[1]<1 || itemp[1]>E->lmesh.nel)
                                        {
                                            fprintf(E->trace.fpt,"ERROR(make_regular)-huh? wierd itemp\n");
                                            fflush(E->trace.fpt);
                                            exit(10);
                                        }
                                }
                            else if ( (ichoice>0) && (ichoice<5) )
                                {
                                    E->trace.regtoel[j][0][iregel]=ichoice;
                                    for (pp=1;pp<=ichoice;pp++)
                                        {
                                            E->trace.regtoel[j][pp][iregel]=itemp[pp];

                                            /*
                                              fprintf(E->trace.fpt,"HH:(%p)  iregel: %d ichoice: %d pp: %d value: %d %d\n",&E->trace.regtoel[j][pp][iregel],iregel,ichoice,pp,itemp[pp],E->trace.regtoel[j][pp][iregel]);
                                            */
                                            if (itemp[pp]<1 || itemp[pp]>E->lmesh.nel)
                                                {
                                                    fprintf(E->trace.fpt,"ERROR(make_regular)-huh? wierd itemp 2 \n");
                                                    fflush(E->trace.fpt);
                                                    exit(10);
                                                }
                                        }
                                }
                            else
                                {
                                    fprintf(E->trace.fpt,"ERROR(make_regular)- should not be here! %d\n",ichoice);
                                    fflush(E->trace.fpt);
                                    exit(10);
                                }
                        }
                }

            /* can now free regnodetoel */

            free (E->trace.regnodetoel[j]);


            /* testing */
            for (kk=1;kk<=E->trace.numregel[j];kk++)
                {
                    if ((E->trace.regtoel[j][0][kk]<-1)||(E->trace.regtoel[j][0][kk]>4))
                        {
                            fprintf(E->trace.fpt,"ERROR(make regular) regtoel ichoice0? %d %d \n",kk,E->trace.regtoel[j][pp][kk]);
                            fflush(E->trace.fpt);
                            exit(10);
                        }
                    for (pp=1;pp<=4;pp++)
                        {
                            if (((E->trace.regtoel[j][pp][kk]<1)&&(E->trace.regtoel[j][pp][kk]!=-33))||(E->trace.regtoel[j][pp][kk]>E->lmesh.nel))
                                {
                                    fprintf(E->trace.fpt,"ERROR(make regular) (%p) regtoel? %d %d(%d) %d\n",&E->trace.regtoel[j][pp][kk],kk,pp,E->trace.regtoel[j][0][kk],E->trace.regtoel[j][pp][kk]);
                                    fflush(E->trace.fpt);
                                    exit(10);
                                }
                        }
                }

        } /* end j */


    fprintf(E->trace.fpt,"Mapping completed (%f seconds)\n",CPU_time0()-start_time);
    fflush(E->trace.fpt);

    parallel_process_sync(E);

    if (E->parallel.me==0) fprintf(stderr,"Mapping completed (%f seconds)\n",CPU_time0()-start_time);
    fflush(stderr);

    /* Print out information regarding regular/real element coverage */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {

            isum=0;
            for (kk=0;kk<=4;kk++) isum=isum+istat_ichoice[j][kk];
            fprintf(E->trace.fpt,"\n\nInformation regarding number of real elements per regular elements\n");
            fprintf(E->trace.fpt," (stats done on regular elements that were used)\n");
            fprintf(E->trace.fpt,"Ichoice is number of real elements touched by a regular element\n");
            fprintf(E->trace.fpt,"  (ichoice=1 is optimal)\n");
            fprintf(E->trace.fpt,"  (if ichoice=0, no elements are touched by regular element)\n");
            fprintf(E->trace.fpt,"Ichoice=0: %f percent\n",(100.0*istat_ichoice[j][0])/(1.0*isum));
            fprintf(E->trace.fpt,"Ichoice=1: %f percent\n",(100.0*istat_ichoice[j][1])/(1.0*isum));
            fprintf(E->trace.fpt,"Ichoice=2: %f percent\n",(100.0*istat_ichoice[j][2])/(1.0*isum));
            fprintf(E->trace.fpt,"Ichoice=3: %f percent\n",(100.0*istat_ichoice[j][3])/(1.0*isum));
            fprintf(E->trace.fpt,"Ichoice=4: %f percent\n",(100.0*istat_ichoice[j][4])/(1.0*isum));

        } /* end j */


    return;
}



/*********  ICHECK COLUMN NEIGHBORS ***************************/
/*                                                            */
/* This function check whether a point is in a neighboring    */
/* column. Neighbor surface element number is returned        */

int icheck_column_neighbors(E,j,nel,x,y,z,rad)
     struct All_variables *E;
     int j,nel;
     double x,y,z,rad;
{

    int ival;
    int neighbor[25];
    int elx,ely,elz;
    int elxz;
    int kk;

    int icheck_element_column();

    /*
      const int number_of_neighbors=24;
    */

    /* maybe faster to only check inner ring */

    const int number_of_neighbors=8;

    elx=E->lmesh.elx;
    ely=E->lmesh.ely;
    elz=E->lmesh.elz;

    elxz=elx*elz;

    /* inner ring */

    neighbor[1]=nel-elxz-elz;
    neighbor[2]=nel-elxz;
    neighbor[3]=nel-elxz+elz;
    neighbor[4]=nel-elz;
    neighbor[5]=nel+elz;
    neighbor[6]=nel+elxz-elz;
    neighbor[7]=nel+elxz;
    neighbor[8]=nel+elxz+elz;

    /* outer ring */

    neighbor[9]=nel+2*elxz-elz;
    neighbor[10]=nel+2*elxz;
    neighbor[11]=nel+2*elxz+elz;
    neighbor[12]=nel+2*elxz+2*elz;
    neighbor[13]=nel+elxz+2*elz;
    neighbor[14]=nel+2*elz;
    neighbor[15]=nel-elxz+2*elz;
    neighbor[16]=nel-2*elxz+2*elz;
    neighbor[17]=nel-2*elxz+elz;
    neighbor[18]=nel-2*elxz;
    neighbor[19]=nel-2*elxz-elz;
    neighbor[20]=nel-2*elxz-2*elz;
    neighbor[21]=nel-elxz-2*elz;
    neighbor[22]=nel-2*elz;
    neighbor[23]=nel+elxz-2*elz;
    neighbor[24]=nel+2*elxz-2*elz;


    for (kk=1;kk<=number_of_neighbors;kk++)
        {

            if ((neighbor[kk]>=1)&&(neighbor[kk]<=E->lmesh.nel))
                {
                    ival=icheck_element_column(E,j,neighbor[kk],x,y,z,rad);
                    if (ival>0)
                        {
                            return neighbor[kk];
                        }
                }
        }

    return -99;
}

/********** ICHECK ALL COLUMNS ********************************/
/*                                                            */
/* This function check all columns until the proper one for   */
/* a point (x,y,z) is found. The surface element is returned  */
/* else -99 if can't be found.                                */

int icheck_all_columns(E,j,x,y,z,rad)
     struct All_variables *E;
     int j;
     double x,y,z,rad;
{

    int icheck;
    int nel;
    int icheck_element_column();

    int elz=E->lmesh.elz;
    int numel=E->lmesh.nel;

    for (nel=elz;nel<=numel;nel=nel+elz)
        {
            icheck=icheck_element_column(E,j,nel,x,y,z,rad);
            if (icheck==1)
                {
                    return nel;
                }
        }


    return -99;
}



/**** WRITE TRACE INSTRUCTIONS ***************/
void write_trace_instructions(E)
     struct All_variables *E;
{
    fprintf(E->trace.fpt,"\nTracing Activated! (proc: %d)\n",E->parallel.me);
    fprintf(E->trace.fpt,"   Allen K. McNamara 12-2003\n\n");

    if (E->trace.ic_method==0)
        {
            fprintf(E->trace.fpt,"Generating New Tracer Array\n");
            fprintf(E->trace.fpt,"Tracers per element: %d\n",E->trace.itperel);
            /* TODO: move
            if (E->composition.ichemical_buoyancy==1)
                {
                    fprintf(E->trace.fpt,"Interface Height: %f\n",E->composition.z_interface);
                }
            */
        }
    if (E->trace.ic_method==1)
        {
            fprintf(E->trace.fpt,"Reading tracer file %s\n",E->trace.tracer_file);
        }
    if (E->trace.ic_method==2)
        {
            fprintf(E->trace.fpt,"Restarting Tracers\n");
        }

    fprintf(E->trace.fpt,"Number of tracer types: %d", E->trace.ntypes);
    if (E->trace.ntypes < 1) {
        fprintf(E->trace.fpt, "Tracer types shouldn't be less than 1\n");
        parallel_process_termination();
    }

    if (E->trace.itracer_advection_scheme==1)
        {
            fprintf(E->trace.fpt,"\nSimple predictor-corrector method used\n");
            fprintf(E->trace.fpt,"(Uses only velocity at to) \n");
            fprintf(E->trace.fpt,"(xf=x0+0.5*dt*(v(x0,y0,z0) + v(xp,yp,zp))\n\n");
        }
    else if (E->trace.itracer_advection_scheme==2)
        {
            fprintf(E->trace.fpt,"\nPredictor-corrector method used\n");
            fprintf(E->trace.fpt,"(Uses only velocity at to and to+dt) \n");
            fprintf(E->trace.fpt,"(xf=x0+0.5*dt*(v(x0,y0,z0,to) + v(xp,yp,zp,to+dt))\n\n");
        }
    else
        {
            fprintf(E->trace.fpt,"Sorry-Other Advection Schemes are Unavailable %d\n",E->trace.itracer_advection_scheme);
            fflush(E->trace.fpt);
            parallel_process_termination();
        }

    if (E->trace.itracer_interpolation_scheme==1)
        {
            fprintf(E->trace.fpt,"\nGreat Circle Projection Interpolation Scheme \n");
        }
    else if (E->trace.itracer_interpolation_scheme==2)
        {
            fprintf(E->trace.fpt,"\nSimple Averaging Interpolation Scheme \n");
            fprintf(E->trace.fpt,"\n(Not that great of a scheme!) \n");

            fprintf(E->trace.fpt,"Sorry-Other Interpolation Schemes are Unavailable\n");
            fflush(E->trace.fpt);
            parallel_process_termination();

        }
    else
        {
            fprintf(E->trace.fpt,"Sorry-Other Interpolation Schemes are Unavailable\n");
            fflush(E->trace.fpt);
            parallel_process_termination();
        }


    /* regular grid stuff */

    fprintf(E->trace.fpt,"Regular Grid-> deltheta: %f delphi: %f\n",
            E->trace.deltheta[0],E->trace.delphi[0]);




    /* more obscure stuff */

    fprintf(E->trace.fpt,"Box Cushion: %f\n",E->trace.box_cushion);
    fprintf(E->trace.fpt,"Number of Basic Quantities: %d\n",
            E->trace.number_of_basic_quantities);
    fprintf(E->trace.fpt,"Number of Extra Quantities: %d\n",
            E->trace.number_of_extra_quantities);
    fprintf(E->trace.fpt,"Total Number of Tracer Quantities: %d\n",
            E->trace.number_of_tracer_quantities);


    /* analytical test */

    if (E->trace.ianalytical_tracer_test==1)
        {
            fprintf(E->trace.fpt,"\n\n ! Analytical Test Being Performed ! \n");
            fprintf(E->trace.fpt,"(some of the above parameters may not be used or applied\n");
            fprintf(E->trace.fpt,"Velocity functions given in main code\n");
            fflush(E->trace.fpt);
        }

    if (E->trace.itracer_warnings==0)
        {
            fprintf(E->trace.fpt,"\n WARNING EXITS ARE TURNED OFF! TURN THEM ON!\n");
            fprintf(stderr,"\n WARNING EXITS ARE TURNED OFF! TURN THEM ON!\n");
            fflush(E->trace.fpt);
            fflush(stderr);
        }

    write_composition_instructions(E);
    return;
}


/************** RESTART TRACERS ******************************************/
/*                                                                       */
/* This function restarts tracers written from previous calculation      */
/* and the tracers are read as seperate files for each processor domain. */

void restart_tracers(E)
     struct All_variables *E;
{

    char output_file[200];
    char input_s[1000];

    int i,j,kk;
    int idum1,ncolumns;
    int numtracers;

    double rdum1;
    double theta,phi,rad;
    double extra[100];
    double x,y,z;

    void initialize_tracer_arrays();
    void sphere_to_cart();
    void keep_in_sphere();

    FILE *fp1;

    if (E->trace.number_of_extra_quantities>99) {
        fprintf(E->trace.fpt,"ERROR(restart_tracers)-increase size of extra[]\n");
        fflush(E->trace.fpt);
        parallel_process_termination();
    }

    sprintf(output_file,"%s.tracer.%d.%d",E->control.old_P_file,E->parallel.me,E->monitor.solution_cycles_init);

    if ( (fp1=fopen(output_file,"r"))==NULL) {
        fprintf(E->trace.fpt,"ERROR(restart tracers)-file not found %s\n",output_file);
        fflush(E->trace.fpt);
        exit(10);
    }

    fprintf(stderr,"Restarting Tracers from %s\n",output_file);
    fflush(stderr);


    for(j=1;j<=E->sphere.caps_per_proc;j++) {
        fgets(input_s,200,fp1);
        sscanf(input_s,"%d %d %d %lf",
               &idum1, &numtracers, &ncolumns, &rdum1);

        /* some error control */
        if (E->trace.number_of_extra_quantities+3 != ncolumns) {
            fprintf(E->trace.fpt,"ERROR(restart tracers)-wrong # of columns\n");
            fflush(E->trace.fpt);
            exit(10);
        }

        /* allocate memory for tracer arrays */

        initialize_tracer_arrays(E,j,numtracers);
        E->trace.ntracers[j]=numtracers;

        for (kk=1;kk<=numtracers;kk++) {
            fgets(input_s,200,fp1);
            if (E->trace.number_of_extra_quantities==0) {
                sscanf(input_s,"%lf %lf %lf\n",&theta,&phi,&rad);
            }
            else if (E->trace.number_of_extra_quantities==1) {
                sscanf(input_s,"%lf %lf %lf %lf\n",&theta,&phi,&rad,&extra[0]);
            }
            /* XXX: if E->trace.number_of_extra_quantities is greater than 1 */
            /* this part has to be changed... */
            else {
                fprintf(E->trace.fpt,"ERROR(restart tracers)-huh?\n");
                fflush(E->trace.fpt);
                exit(10);
            }

            sphere_to_cart(E,theta,phi,rad,&x,&y,&z);

            /* it is possible that if on phi=0 boundary, significant digits can push phi over 2pi */

            keep_in_sphere(E,&x,&y,&z,&theta,&phi,&rad);

            E->trace.basicq[j][0][kk]=theta;
            E->trace.basicq[j][1][kk]=phi;
            E->trace.basicq[j][2][kk]=rad;
            E->trace.basicq[j][3][kk]=x;
            E->trace.basicq[j][4][kk]=y;
            E->trace.basicq[j][5][kk]=z;

            for (i=0; i<E->trace.number_of_extra_quantities; i++)
                E->trace.extraq[j][i][kk]=extra[i];

        }

        fprintf(E->trace.fpt,"Read %d tracers from file %s\n",numtracers,output_file);
        fflush(E->trace.fpt);

    }


    return;
}

/************** MAKE TRACER ARRAY ********************************/
/* Here, each cap will generate tracers somewhere                */
/* in the sphere - check if its in this cap  - then check radial */

static void make_tracer_array(struct All_variables *E)
{

    int kk;
    int tracers_cap;
    int j;
    int ival;
    int number_of_tries=0;
    int max_tries;

    double x,y,z;
    double theta,phi,rad;
    double dmin,dmax;
    double random1,random2,random3;
    double processor_fraction;

    void cart_to_sphere();
    void keep_in_sphere();
    void initialize_tracer_arrays();

    int icheck_cap();

    if (E->parallel.me==0) fprintf(stderr,"Making Tracer Array\n");
    fflush(stderr);


    for (j=1;j<=E->sphere.caps_per_proc;j++) {

        processor_fraction=( ( pow(E->sx[j][3][E->lmesh.noz],3.0)-pow(E->sx[j][3][1],3.0))/
                             (pow(E->sphere.ro,3.0)-pow(E->sphere.ri,3.0)));
        tracers_cap=E->mesh.nel*E->trace.itperel*processor_fraction;
        /*
          fprintf(stderr,"AA: proc frac: %f (%d) %d %d %f %f\n",processor_fraction,tracers_cap,E->lmesh.nel,E->parallel.nprocz, E->sx[j][3][E->lmesh.noz],E->sx[j][3][1]);
        */

        fprintf(E->trace.fpt,"\nGenerating %d Tracers\n",tracers_cap);

        initialize_tracer_arrays(E,j,tracers_cap);


        /* Tracers are placed randomly in cap */
        /* (intentionally using rand() instead of srand() )*/

        dmin=-1.0*E->sphere.ro;
        dmax=E->sphere.ro;

        while (E->trace.ntracers[j]<tracers_cap) {

            number_of_tries++;
            max_tries=500*tracers_cap;

            if (number_of_tries>max_tries) {
                fprintf(E->trace.fpt,"Error(make_tracer_array)-too many tries?\n");
                fprintf(E->trace.fpt,"%d %d %d\n",max_tries,number_of_tries,RAND_MAX);
                fflush(E->trace.fpt);
                exit(10);
            }


            random1=(1.0*rand())/(1.0*RAND_MAX);
            random2=(1.0*rand())/(1.0*RAND_MAX);
            random3=(1.0*rand())/(1.0*RAND_MAX);

            x=dmin+random1*(dmax-dmin);
            y=dmin+random2*(dmax-dmin);
            z=dmin+random3*(dmax-dmin);

            /* first check if within shell */

            rad=sqrt(x*x+y*y+z*z);

            if (rad>=E->sx[j][3][E->lmesh.noz]) continue;
            if (rad<E->sx[j][3][1]) continue;


            /* check if in current cap */

            ival=icheck_cap(E,0,x,y,z,rad);

            if (ival!=1) continue;

            /* Made it, so record tracer information */

            cart_to_sphere(E,x,y,z,&theta,&phi,&rad);

            keep_in_sphere(E,&x,&y,&z,&theta,&phi,&rad);

            E->trace.ntracers[j]++;
            kk=E->trace.ntracers[j];

            E->trace.basicq[j][0][kk]=theta;
            E->trace.basicq[j][1][kk]=phi;
            E->trace.basicq[j][2][kk]=rad;
            E->trace.basicq[j][3][kk]=x;
            E->trace.basicq[j][4][kk]=y;
            E->trace.basicq[j][5][kk]=z;

        } /* end while */


    }/* end j */

    fprintf(stderr,"DONE Making Tracer Array (%d)\n",E->parallel.me);
    fflush(stderr);

    return;
}

/******** READ TRACER ARRAY *********************************************/
/*                                                                      */
/* This function reads tracers from input file.                         */
/* All processors read the same input file, then sort out which ones    */
/* belong.                                                              */

void read_tracer_file(E)
     struct All_variables *E;
{

    char input_s[1000];

    int number_of_tracers;
    int kk;
    int icheck;
    int iestimate;
    int icushion;
    int j;

    int icheck_cap();
    int icheck_processor_shell();
    int isum_tracers();
    void initialize_tracer_arrays();
    void keep_in_sphere();
    void sphere_to_cart();
    void cart_to_sphere();
    void expand_tracer_arrays();

    double x,y,z;
    double theta,phi,rad;
    double rdum1,rdum2,rdum3;

    FILE *fptracer;

    fptracer=fopen(E->trace.tracer_file,"r");
    fprintf(E->trace.fpt,"Opening %s\n",E->trace.tracer_file);

    fgets(input_s,200,fptracer);
    sscanf(input_s,"%d",&number_of_tracers);
    fprintf(E->trace.fpt,"%d Tracers in file \n",number_of_tracers);

    /* initially size tracer arrays to number of tracers divided by processors */

    icushion=100;

    iestimate=number_of_tracers/E->parallel.nproc + icushion;

    for (j=1;j<=E->sphere.caps_per_proc;j++) {

        initialize_tracer_arrays(E,j,iestimate);

        for (kk=1;kk<=number_of_tracers;kk++) {
            fgets(input_s,200,fptracer);
            sscanf(input_s,"%lf %lf %lf",&rdum1,&rdum2,&rdum3);

            theta=rdum1;
            phi=rdum2;
            rad=rdum3;
            sphere_to_cart(E,theta,phi,rad,&x,&y,&z);


            /* make sure theta, phi is in range, and radius is within bounds */

            keep_in_sphere(E,&x,&y,&z,&theta,&phi,&rad);

            /* check whether tracer is within processor domain */

            icheck=1;
            if (E->parallel.nprocz>1) icheck=icheck_processor_shell(E,j,rad);
            if (icheck!=1) continue;

            icheck=icheck_cap(E,0,x,y,z,rad);
            if (icheck==0) continue;

            /* if still here, tracer is in processor domain */


            E->trace.ntracers[j]++;

            if (E->trace.ntracers[j]>=(E->trace.max_ntracers[j]-5)) expand_tracer_arrays(E,j);

            E->trace.basicq[j][0][E->trace.ntracers[j]]=theta;
            E->trace.basicq[j][1][E->trace.ntracers[j]]=phi;
            E->trace.basicq[j][2][E->trace.ntracers[j]]=rad;
            E->trace.basicq[j][3][E->trace.ntracers[j]]=x;
            E->trace.basicq[j][4][E->trace.ntracers[j]]=y;
            E->trace.basicq[j][5][E->trace.ntracers[j]]=z;

        } /* end kk, number of tracers */

        fprintf(E->trace.fpt,"Number of tracers in this cap is: %d\n",
                E->trace.ntracers[j]);

    } /* end j */

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


/********** CART TO SPHERE ***********************/
void cart_to_sphere(E,x,y,z,theta,phi,rad)
     struct All_variables *E;
     double x,y,z;
     double *theta,*phi,*rad;
{

    double temp;
    double myatan();

    temp=x*x+y*y;

    *rad=sqrt(temp+z*z);
    *theta=atan2(sqrt(temp),z);
    *phi=myatan(y,x);


    return;
}

/********** SPHERE TO CART ***********************/
void sphere_to_cart(E,theta,phi,rad,x,y,z)
     struct All_variables *E;
     double theta,phi,rad;
     double *x,*y,*z;
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

int icheck_that_processor_shell(E,j,nprocessor,rad)
     struct All_variables *E;
     int j;
     int nprocessor;
     double rad;
{
    int icheck_processor_shell();
    int me = E->parallel.me;

    /* nprocessor is right on top of me */
    if (nprocessor == me+1) {
        if (icheck_processor_shell(E, j, rad) == 0) return 1;
        else return 0;
    }

    /* nprocessor is right on bottom of me */
    if (nprocessor == me-1) {
        if (icheck_processor_shell(E, j, rad) == -99) return 1;
        else return 0;
    }

    /* Shouldn't be here */
    fprintf(E->trace.fpt, "Should not be here\n");
    fprintf(E->trace.fpt, "Error(check_shell) nprocessor: %d, radius: %f\n",
            nprocessor, rad);
    fflush(E->trace.fpt);
    exit(10);
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

int icheck_processor_shell(E,j,rad)
     struct All_variables *E;
     double rad;
     int j;

{

    const int noz = E->lmesh.noz;
    const int nprocz = E->parallel.nprocz;
    double top_r, bottom_r;

    if (nprocz==1) return 1;

    top_r = E->sx[j][3][noz];
    bottom_r = E->sx[j][3][1];

    /* First check bottom */

    if (rad<bottom_r) return -99;


    /* Check top */

    if (rad<top_r) return 1;

    /* top processor */

    if ( (rad<=top_r) && (E->parallel.me_loc[3]==nprocz-1) ) return 1;

    /* If here, means point is above processor */
    return 0;
}

/******* ICHECK ELEMENT *************************************/
/*                                                          */
/* This function serves to determine if a point lies within */
/* a given element                                          */

int icheck_element(E,j,nel,x,y,z,rad)
     struct All_variables *E;
     int nel,j;
     double x,y,z,rad;
{

    int icheck;
    int icheck_element_column();
    int icheck_shell();

    icheck=icheck_shell(E,nel,rad);
    if (icheck==0)
        {
            return 0;
        }

    icheck=icheck_element_column(E,j,nel,x,y,z,rad);
    if (icheck==0)
        {
            return 0;
        }


    return 1;
}

/********  ICHECK SHELL ************************************/
/*                                                         */
/* This function serves to check whether a point lies      */
/* within the proper radial shell of a given element       */
/* note: j set to 1; shouldn't depend on cap               */

int icheck_shell(E,nel,rad)
     struct All_variables *E;
     int nel;
     double rad;
{

    int ival;
    int ibottom_node;
    int itop_node;

    double bottom_rad;
    double top_rad;


    ibottom_node=E->ien[1][nel].node[1];
    itop_node=E->ien[1][nel].node[5];

    bottom_rad=E->sx[1][3][ibottom_node];
    top_rad=E->sx[1][3][itop_node];

    ival=0;
    if ((rad>=bottom_rad)&&(rad<top_rad)) ival=1;

    return ival;
}

/********  ICHECK ELEMENT COLUMN ****************************/
/*                                                          */
/* This function serves to determine if a point lies within */
/* a given element's column                                 */

int icheck_element_column(E,j,nel,x,y,z,rad)
     struct All_variables *E;
     int nel,j;
     double x,y,z,rad;
{

    double test_point[4];
    double rnode[5][10];

    int lev = E->mesh.levmax;
    int ival;
    int kk;
    int node;
    int icheck_bounds();


    E->trace.istat_elements_checked++;

    /* surface coords of element nodes */

    for (kk=1;kk<=4;kk++)
        {

            node=E->ien[j][nel].node[kk+4];

            rnode[kk][1]=E->x[j][1][node];
            rnode[kk][2]=E->x[j][2][node];
            rnode[kk][3]=E->x[j][3][node];

            rnode[kk][4]=E->sx[j][1][node];
            rnode[kk][5]=E->sx[j][2][node];

            rnode[kk][6]=E->SinCos[lev][j][2][node]; /* cos(theta) */
            rnode[kk][7]=E->SinCos[lev][j][0][node]; /* sin(theta) */
            rnode[kk][8]=E->SinCos[lev][j][3][node]; /* cos(phi) */
            rnode[kk][9]=E->SinCos[lev][j][1][node]; /* sin(phi) */

        }

    /* test_point - project to outer radius */

    test_point[1]=x/rad;
    test_point[2]=y/rad;
    test_point[3]=z/rad;

    ival=icheck_bounds(E,test_point,rnode[1],rnode[2],rnode[3],rnode[4]);


    return ival;
}


/********* ICHECK CAP ***************************************/
/*                                                          */
/* This function serves to determine if a point lies within */
/* a given cap                                              */
/*                                                          */
int icheck_cap(E,icap,x,y,z,rad)
     struct All_variables *E;
     int icap;
     double x,y,z,rad;
{

    double test_point[4];
    double rnode[5][10];

    int ival;
    int kk;
    int icheck_bounds();

    /* surface coords of cap nodes */


    for (kk=1;kk<=4;kk++)
        {

            rnode[kk][1]=E->trace.xcap[icap][kk];
            rnode[kk][2]=E->trace.ycap[icap][kk];
            rnode[kk][3]=E->trace.zcap[icap][kk];
            rnode[kk][4]=E->trace.theta_cap[icap][kk];
            rnode[kk][5]=E->trace.phi_cap[icap][kk];
            rnode[kk][6]=E->trace.cos_theta[icap][kk];
            rnode[kk][7]=E->trace.sin_theta[icap][kk];
            rnode[kk][8]=E->trace.cos_phi[icap][kk];
            rnode[kk][9]=E->trace.sin_phi[icap][kk];
        }


    /* test_point - project to outer radius */

    test_point[1]=x/rad;
    test_point[2]=y/rad;
    test_point[3]=z/rad;

    ival=icheck_bounds(E,test_point,rnode[1],rnode[2],rnode[3],rnode[4]);


    return ival;
}

/***** ICHECK BOUNDS ******************************/
/*                                                */
/* This function check if a test_point is bounded */
/* by 4 nodes                                     */
/* This is done by:                               */
/* 1) generate vectors from node to node          */
/* 2) generate vectors from each node to point    */
/*    in question                                 */
/* 3) for each node, take cross product of vector */
/*    pointing to it from previous node and       */
/*    vector from node to point in question       */
/* 4) Find radial components of all the cross     */
/*    products.                                   */
/* 5) If all radial components are positive,      */
/*    point is bounded by the 4 nodes             */
/* 6) If some radial components are negative      */
/*    point is on a boundary - adjust it an       */
/*    epsilon amount for this analysis only       */
/*    which will force it to lie in one element   */
/*    or cap                                      */

int icheck_bounds(E,test_point,rnode1,rnode2,rnode3,rnode4)
     struct All_variables *E;
     double *test_point;
     double *rnode1;
     double *rnode2;
     double *rnode3;
     double *rnode4;
{

    int number_of_tries=0;
    int icheck;

    double v12[4];
    double v23[4];
    double v34[4];
    double v41[4];
    double v1p[4];
    double v2p[4];
    double v3p[4];
    double v4p[4];
    double cross1[4];
    double cross2[4];
    double cross3[4];
    double cross4[4];
    double rad1,rad2,rad3,rad4;
    double theta, phi;
    double tiny, eps;
    double x,y,z;

    double findradial();
    void makevector();
    void crossit();
    double myatan();

    /* make vectors from node to node */

    makevector(v12,rnode2[1],rnode2[2],rnode2[3],rnode1[1],rnode1[2],rnode1[3]);
    makevector(v23,rnode3[1],rnode3[2],rnode3[3],rnode2[1],rnode2[2],rnode2[3]);
    makevector(v34,rnode4[1],rnode4[2],rnode4[3],rnode3[1],rnode3[2],rnode3[3]);
    makevector(v41,rnode1[1],rnode1[2],rnode1[3],rnode4[1],rnode4[2],rnode4[3]);

 try_again:

    number_of_tries++;

    /* make vectors from test point to node */

    makevector(v1p,test_point[1],test_point[2],test_point[3],rnode1[1],rnode1[2],rnode1[3]);
    makevector(v2p,test_point[1],test_point[2],test_point[3],rnode2[1],rnode2[2],rnode2[3]);
    makevector(v3p,test_point[1],test_point[2],test_point[3],rnode3[1],rnode3[2],rnode3[3]);
    makevector(v4p,test_point[1],test_point[2],test_point[3],rnode4[1],rnode4[2],rnode4[3]);

    /* Calculate cross products */

    crossit(cross2,v12,v2p);
    crossit(cross3,v23,v3p);
    crossit(cross4,v34,v4p);
    crossit(cross1,v41,v1p);

    /* Calculate radial component of cross products */

    rad1=findradial(E,cross1,rnode1[6],rnode1[7],rnode1[8],rnode1[9]);
    rad2=findradial(E,cross2,rnode2[6],rnode2[7],rnode2[8],rnode2[9]);
    rad3=findradial(E,cross3,rnode3[6],rnode3[7],rnode3[8],rnode3[9]);
    rad4=findradial(E,cross4,rnode4[6],rnode4[7],rnode4[8],rnode4[9]);

    /*  Check if any radial components is zero (along a boundary), adjust if so */
    /*  Hopefully, this doesn't happen often, may be expensive                  */

    tiny=1e-15;
    eps=1e-6;

    if (number_of_tries>3)
        {
            fprintf(E->trace.fpt,"Error(icheck_bounds)-too many tries\n");
            fprintf(E->trace.fpt,"Rads: %f %f %f %f\n",rad1,rad2,rad3,rad4);
            fprintf(E->trace.fpt,"Test Point: %f %f %f  \n",test_point[1],test_point[2],test_point[3]);
            fprintf(E->trace.fpt,"Nodal points: 1: %f %f %f\n",rnode1[1],rnode1[2],rnode1[3]);
            fprintf(E->trace.fpt,"Nodal points: 2: %f %f %f\n",rnode2[1],rnode2[2],rnode2[3]);
            fprintf(E->trace.fpt,"Nodal points: 3: %f %f %f\n",rnode3[1],rnode3[2],rnode3[3]);
            fprintf(E->trace.fpt,"Nodal points: 4: %f %f %f\n",rnode4[1],rnode4[2],rnode4[3]);
            fflush(E->trace.fpt);
            exit(10);
        }

    if (fabs(rad1)<=tiny||fabs(rad2)<=tiny||fabs(rad3)<=tiny||fabs(rad4)<=tiny)
        {
            x=test_point[1];
            y=test_point[2];
            z=test_point[3];
            theta=myatan(sqrt(x*x+y*y),z);
            phi=myatan(y,x);

            if (theta<=M_PI/2.0)
                {
                    theta=theta+eps;
                }
            else
                {
                    theta=theta-eps;
                }
            phi=phi+eps;
            x=sin(theta)*cos(phi);
            y=sin(theta)*sin(phi);
            z=cos(theta);
            test_point[1]=x;
            test_point[2]=y;
            test_point[3]=z;

            number_of_tries++;
            goto try_again;

        }

    icheck=0;
    if (rad1>0.0&&rad2>0.0&&rad3>0.0&&rad4>0.0) icheck=1;

    /*
      fprintf(stderr,"%d: icheck: %d\n",E->parallel.me,icheck);
      fprintf(stderr,"%d: rads: %f %f %f %f\n",E->parallel.me,rad1,rad2,rad3,rad4);
    */

    return icheck;

}

/****************************************************************************/
/* FINDRADIAL                                                              */
/*                                                                          */
/* This function finds the radial component of a Cartesian vector           */


double findradial(E,vec,cost,sint,cosf,sinf)
     struct All_variables *E;
     double *vec;
     double cost,sint,cosf,sinf;
{
    double radialparti,radialpartj,radialpartk;
    double radial;

    radialparti=vec[1]*sint*cosf;
    radialpartj=vec[2]*sint*sinf;
    radialpartk=vec[3]*cost;

    radial=radialparti+radialpartj+radialpartk;


    return radial;
}


/******************MAKEVECTOR*********************************************************/

void makevector(vec,xf,yf,zf,x0,y0,z0)
     double *vec;
     double xf,yf,zf,x0,y0,z0;
{

    vec[1]=xf-x0;
    vec[2]=yf-y0;
    vec[3]=zf-z0;


    return;
}

/********************CROSSIT********************************************************/

void crossit(cross,A,B)
     double *cross;
     double *A;
     double *B;
{

    cross[1]=A[2]*B[3]-A[3]*B[2];
    cross[2]=A[3]*B[1]-A[1]*B[3];
    cross[3]=A[1]*B[2]-A[2]*B[1];


    return;
}


/************ FIX RADIUS ********************************************/
/* This function moves particles back in bounds if they left     */
/* during advection                                              */

static void fix_radius(struct All_variables *E,
                       double *radius, double *theta, double *phi,
                       double *x, double *y, double *z)
{
    double sint,cost,sinf,cosf,rad;
    double max_radius, min_radius;

    max_radius = E->sphere.ro - E->trace.box_cushion;
    min_radius = E->sphere.ri + E->trace.box_cushion;

    if (*radius > max_radius) {
        *radius=max_radius;
        rad=max_radius;
        cost=cos(*theta);
        sint=sqrt(1.0-cost*cost);
        cosf=cos(*phi);
        sinf=sin(*phi);
        *x=rad*sint*cosf;
        *y=rad*sint*sinf;
        *z=rad*cost;
    }
    if (*radius < min_radius) {
        *radius=min_radius;
        rad=min_radius;
        cost=cos(*theta);
        sint=sqrt(1.0-cost*cost);
        cosf=cos(*phi);
        sinf=sin(*phi);
        *x=rad*sint*cosf;
        *y=rad*sint*sinf;
        *z=rad*cost;
    }

    return;
}


/******************************************************************/
/* FIX PHI                                                        */
/*                                                                */
/* This function constrains the value of phi to be                */
/* between 0 and 2 PI                                             */
/*                                                                */

static void fix_phi(double *phi)
{
    const double two_pi=2.0*M_PI;

    double d2 = floor(*phi / two_pi);

    *phi -= two_pi * d2;

    return;
}

/******************************************************************/
/* FIX THETA PHI                                                  */
/*                                                                */
/* This function constrains the value of theta to be              */
/* between 0 and  PI, and                                         */
/* this function constrains the value of phi to be                */
/* between 0 and 2 PI                                             */
/*                                                                */
static void fix_theta_phi(double *theta, double *phi)
{
    const double two_pi=2.0*M_PI;
    double d, d2;

    d = floor(*theta/M_PI);

    *theta -= M_PI * d;
    *phi += M_PI * d;

    d2 = floor(*phi / two_pi);

    *phi -= two_pi * d2;

    return;
}

/********** IGET ELEMENT *****************************************/
/*                                                               */
/* This function returns the the real element for a given point. */
/* Returns -99 in not in this cap.                               */
/* iprevious_element, if known, is the last known element. If    */
/* it is not known, input a negative number.                     */

int iget_element(E,j,iprevious_element,x,y,z,theta,phi,rad)
     struct All_variables *E;
     int j;
     int iprevious_element;
     double x,y,z;
     double theta,phi,rad;
{

    int iregel;
    int iel;
    int ntheta,nphi;
    int ival;
    int ichoice;
    int kk;
    int ineighbor;
    int icorner[5];
    int elx,ely,elz,elxz;
    int ifinal_iel;
    int nelem;

    int iget_regel();
    int iquick_element_column_search();
    int icheck_cap();
    int icheck_regular_neighbors();
    int iget_radial_element();

    elx=E->lmesh.elx;
    ely=E->lmesh.ely;
    elz=E->lmesh.elz;


    ntheta=0;
    nphi=0;

    /* check the radial range */
    if (E->parallel.nprocz>1)
        {
            ival=icheck_processor_shell(E,j,rad);
            if (ival!=1) return -99;
        }

    /* First check previous element */

    /* do quick search to see if element can be easily found. */
    /* note that element may still be out of this cap, but    */
    /* it is probably fast to do a quick search before        */
    /* checking cap                                           */


    /* get regular element number */

    iregel=iget_regel(E,j,theta,phi,&ntheta,&nphi);
    if (iregel<=0)
        {
            return -99;
        }


    /* AKMA put safety here or in make grid */

    if (E->trace.regtoel[j][0][iregel]==0)
        {
            iel=E->trace.regtoel[j][1][iregel];
            goto foundit;
        }

    /* first check previous element */

    if (iprevious_element>0)
        {
            ival=icheck_element_column(E,j,iprevious_element,x,y,z,rad);
            if (ival==1)
                {
                    iel=iprevious_element;
                    goto foundit;
                }
        }

    /* Check all regular mapping choices */

    ichoice=0;
    if (E->trace.regtoel[j][0][iregel]>0)
        {

            ichoice=E->trace.regtoel[j][0][iregel];
            for (kk=1;kk<=ichoice;kk++)
                {
                    nelem=E->trace.regtoel[j][kk][iregel];

                    if (nelem!=iprevious_element)
                        {
                            ival=icheck_element_column(E,j,nelem,x,y,z,rad);
                            if (ival==1)
                                {
                                    iel=nelem;
                                    goto foundit;
                                }

                        }
                }
        }

    /* If here, it means that tracer could not be found quickly with regular element map */

    /* First check previous element neighbors */

    if (iprevious_element>0)
        {
            iel=icheck_column_neighbors(E,j,iprevious_element,x,y,z,rad);
            if (iel>0)
                {
                    goto foundit;
                }
        }

    /* check if still in cap */

    ival=icheck_cap(E,0,x,y,z,rad);
    if (ival==0)
        {
            return -99;
        }

    /* if here, still in cap (hopefully, without a doubt) */

    /* check cap corners (they are sometimes tricky) */

    elxz=elx*elz;
    icorner[1]=elz;
    icorner[2]=elxz;
    icorner[3]=elxz*(ely-1)+elz;
    icorner[4]=elxz*ely;
    for (kk=1;kk<=4;kk++)
        {
            ival=icheck_element_column(E,j,icorner[kk],x,y,z,rad);
            if (ival>0)
                {
                    iel=icorner[kk];
                    goto foundit;
                }
        }


    /* if previous element is not known, check neighbors of those tried in iquick... */

    if (iprevious_element<0)
        {
            if (ichoice>0)
                {
                    for (kk=1;kk<=ichoice;kk++)
                        {
                            ineighbor=E->trace.regtoel[j][kk][iregel];
                            iel=icheck_column_neighbors(E,j,ineighbor,x,y,z,rad);
                            if (iel>0)
                                {
                                    goto foundit;
                                }
                        }
                }

            /* Decided to remove this part - not really needed and  complicated */
            /*
              else
              {
              iel=icheck_regular_neighbors(E,j,ntheta,nphi,x,y,z,theta,phi,rad);
              if (iel>0)
              {
              goto foundit;
              }
              }
            */
        }

    /* As a last resort, check all element columns */

    E->trace.istat1++;

    iel=icheck_all_columns(E,j,x,y,z,rad);

    /*
      fprintf(E->trace.fpt,"WARNING(iget_element)-doing a full search!\n");
      fprintf(E->trace.fpt,"  Most often means tracers have moved more than 1 element away\n");
      fprintf(E->trace.fpt,"  or regular element resolution is way too low.\n");
      fprintf(E->trace.fpt,"  COLUMN: %d \n",iel);
      fprintf(E->trace.fpt,"  PREVIOUS ELEMENT: %d \n",iprevious_element);
      fprintf(E->trace.fpt,"  x,y,z,theta,phi,rad: %f %f %f   %f %f %f\n",x,y,z,theta,phi,rad);
      fflush(E->trace.fpt);
      if (E->trace.itracer_warnings==1) exit(10);
    */

    if (E->trace.istat1%100==0)
        {
            fprintf(E->trace.fpt,"Checked all elements %d times already this turn\n",E->trace.istat1);
            fprintf(stderr,"Checked all elements %d times already this turn\n",E->trace.istat1);
            fflush(E->trace.fpt);
            fflush(stderr);
        }
    if (iel>0)
        {
            goto foundit;
        }


    /* if still here, there is a problem */

    fprintf(E->trace.fpt,"Error(iget_element) - element not found\n");
    fprintf(E->trace.fpt,"x,y,z,theta,phi,iregel %f %f %f %f %f %d\n",
            x,y,z,theta,phi,iregel);
    fflush(E->trace.fpt);
    exit(10);

 foundit:

    /* find radial element */

    ifinal_iel=iget_radial_element(E,j,iel,rad);

    return ifinal_iel;
}


/***** IGET RADIAL ELEMENT ***********************************/
/*                                                           */
/* This function returns the proper radial element, given    */
/* an element (iel) from the column.                         */

int iget_radial_element(E,j,iel,rad)
     struct All_variables *E;
     int j,iel;
     double rad;
{

    int elz=E->lmesh.elz;
    int ibottom_element;
    int iradial_element;
    int node;
    int kk;
    int idum;

    double top_rad;

    /* first project to the lowest element in the column */

    idum=(iel-1)/elz;
    ibottom_element=idum*elz+1;

    iradial_element=ibottom_element;

    for (kk=1;kk<=elz;kk++)
        {

            node=E->ien[j][iradial_element].node[8];
            top_rad=E->sx[j][3][node];

            if (rad<top_rad) goto found_it;

            iradial_element++;

        } /* end kk */


    /* should not be here */

    fprintf(E->trace.fpt,"Error(iget_radial_element)-out of range %f %d %d %d\n",rad,j,iel,ibottom_element);
    fflush(E->trace.fpt);
    exit(10);

 found_it:

    return iradial_element;
}

/****** ICHECK REGULAR NEIGHBORS *****************************/
/*                                                           */
/* This function searches the regular element neighborhood.  */

/* This function is no longer used!                          */

int icheck_regular_neighbors(E,j,ntheta,nphi,x,y,z,theta,phi,rad)
     struct All_variables *E;
     int j,ntheta,nphi;
     double x,y,z;
     double theta,phi,rad;
{

    int new_ntheta,new_nphi;
    int kk,pp;
    int iregel;
    int ival;
    int imap[5];
    int ichoice;
    int irange;

    int iquick_element_column_search();

    fprintf(E->trace.fpt,"ERROR(icheck_regular_neighbors)-this subroutine is no longer used !\n");
    fflush(E->trace.fpt);
    exit(10);

    irange=2;

    for (kk=-irange;kk<=irange;kk++)
        {
            for (pp=-irange;pp<=irange;pp++)
                {
                    new_ntheta=ntheta+kk;
                    new_nphi=nphi+pp;
                    if ( (new_ntheta>0)&&(new_ntheta<=E->trace.numtheta[j])&&(new_nphi>0)&&(new_nphi<=E->trace.numphi[j]) )
                        {
                            iregel=new_ntheta+(new_nphi-1)*E->trace.numtheta[j];
                            if ((iregel>0) && (iregel<=E->trace.numregel[j]))
                                {
                                    ival=iquick_element_column_search(E,j,iregel,new_ntheta,new_nphi,x,y,z,theta,phi,rad,imap,&ichoice);
                                    if (ival>0) return ival;
                                }
                        }
                }
        }


    return -99;
}





/****** IQUICK ELEMENT SEARCH *****************************/
/*                                                        */
/* This function does a quick regular to real element     */
/* map check. Element number, if found, is returned.      */
/* Otherwise, -99 is returned.                            */
/* Pointers to imap and ichoice are used because they may */
/* prove to be convenient.                                */
/* This routine is no longer used                         */

int iquick_element_column_search(E,j,iregel,ntheta,nphi,x,y,z,theta,phi,rad,imap,ich)
     struct All_variables *E;
     int j,iregel;
     int ntheta,nphi;
     double x,y,z,theta,phi,rad;
     int *imap;
     int *ich;
{

    int iregnode[5];
    int kk,pp;
    int nel,ival;
    int ichoice;
    int icount;
    int itemp1;
    int itemp2;

    int icheck_element_column();

    fprintf(E->trace.fpt,"ERROR(iquick element)-this routine is no longer used!\n");
    fflush(E->trace.fpt);
    exit(10);

    /* REMOVE*/
    /*
      ichoice=*ich;

      fprintf(E->trace.fpt,"AA: ichoice: %d\n",ichoice);
      fflush(E->trace.fpt);
    */

    /* find regular nodes on regular element */

    /*
      iregnode[1]=iregel+(nphi-1);
      iregnode[2]=iregel+nphi;
      iregnode[3]=iregel+nphi+E->trace.numtheta[j]+1;
      iregnode[4]=iregel+nphi+E->trace.numtheta[j];
    */

    itemp1=iregel+nphi;
    itemp2=itemp1+E->trace.numtheta[j];

    iregnode[1]=itemp1-1;
    iregnode[2]=itemp1;
    iregnode[3]=itemp2+1;
    iregnode[4]=itemp2;

    for (kk=1;kk<=4;kk++)
        {
            if ((iregnode[kk]<1) || (iregnode[kk]>E->trace.numregnodes[j]) )
                {
                    fprintf(E->trace.fpt,"ERROR(iquick)-weird regnode %d\n",iregnode[kk]);
                    fflush(E->trace.fpt);
                    exit(10);
                }
        }

    /* find number of choices */

    ichoice=0;
    icount=0;
    for (kk=1;kk<=4;kk++)
        {
            if (E->trace.regnodetoel[j][iregnode[kk]]<=0) goto next_corner;

            icount++;
            for (pp=1;pp<=(kk-1);pp++)
                {
                    if (E->trace.regnodetoel[j][iregnode[kk]]==E->trace.regnodetoel[j][iregnode[pp]]) goto next_corner;
                }
            ichoice++;
            imap[ichoice]=E->trace.regnodetoel[j][iregnode[kk]];


        next_corner:
            ;
        } /* end kk */

    *ich=ichoice;

    /* statistical counter */

    E->trace.istat_ichoice[j][ichoice]++;

    if (ichoice==0) return -99;

    /* Here, no check is performed if all 4 corners */
    /* lie within a given element.                  */
    /* It may be possible (not sure) but unlikely   */
    /* that the tracer is still not in that element */

    /* Decided to comment this out. */
    /* May not be valid for large regular grids. */
    /*
     */
    /* AKMA */

    if ((ichoice==1)&&(icount==4)) return imap[1];

    /* check others */

    for (kk=1;kk<=ichoice;kk++)
        {
            nel=imap[kk];
            ival=icheck_element_column(E,j,nel,x,y,z,rad);
            if (ival>0) return nel;
        }

    /* if still here, no element was found */

    return -99;
}


/*********** IGET REGEL ******************************************/
/*                                                               */
/* This function returns the regular element in which a point    */
/* exists. If not found, returns -99.                            */
/* npi and ntheta are modified for later use                     */

int iget_regel(E,j,theta,phi,ntheta,nphi)
     struct All_variables *E;
     int j;
     double theta, phi;
     int *ntheta;
     int *nphi;
{

    int iregel;
    int idum;

    double rdum;

    /* first check whether theta is in range */

    if (theta<E->trace.thetamin[j]) return -99;
    if (theta>E->trace.thetamax[j]) return -99;

    /* get ntheta, nphi on regular mesh */

    rdum=theta-E->trace.thetamin[j];
    idum=rdum/E->trace.deltheta[j];
    *ntheta=idum+1;

    rdum=phi-E->trace.phimin[j];
    idum=rdum/E->trace.delphi[j];
    *nphi=idum+1;

    iregel=*ntheta+(*nphi-1)*E->trace.numtheta[j];

    /* check range to be sure */

    if (iregel>E->trace.numregel[j]) return -99;
    if (iregel<1) return -99;

    return iregel;
}


/****** EXPAND TRACER ARRAYS *****************************************/

void expand_tracer_arrays(E,j)
     struct All_variables *E;
     int j;

{

    int inewsize;
    int kk;
    int icushion;

    /* expand basicq and ielement by 20% */

    icushion=100;

    inewsize=E->trace.max_ntracers[j]+E->trace.max_ntracers[j]/5+icushion;

    if ((E->trace.ielement[j]=(int *)realloc(E->trace.ielement[j],inewsize*sizeof(int)))==NULL)
        {
            fprintf(E->trace.fpt,"ERROR(expand tracer arrays )-no memory (ielement)\n");
            fflush(E->trace.fpt);
            exit(10);
        }

    for (kk=0;kk<=((E->trace.number_of_basic_quantities)-1);kk++)
        {
            if ((E->trace.basicq[j][kk]=(double *)realloc(E->trace.basicq[j][kk],inewsize*sizeof(double)))==NULL)
                {
                    fprintf(E->trace.fpt,"ERROR(expand tracer arrays )-no memory (%d)\n",kk);
                    fflush(E->trace.fpt);
                    exit(10);
                }
        }

    for (kk=0;kk<=((E->trace.number_of_extra_quantities)-1);kk++)
        {
            if ((E->trace.extraq[j][kk]=(double *)realloc(E->trace.extraq[j][kk],inewsize*sizeof(double)))==NULL)
                {
                    fprintf(E->trace.fpt,"ERROR(expand tracer arrays )-no memory 78 (%d)\n",kk);
                    fflush(E->trace.fpt);
                    exit(10);
                }
        }


    fprintf(E->trace.fpt,"Expanding physical memory of ielement, basicq, and extraq to %d from %d\n",
            inewsize,E->trace.max_ntracers[j]);

    E->trace.max_ntracers[j]=inewsize;

    return;
}

/****************************************************************/
/* DEFINE UV SPACE                                              */
/*                                                              */
/* This function defines nodal points as orthodrome coordinates */
/* u and v.  In uv space, great circles form straight lines.    */
/* This is used for interpolation method 1                      */
/* UV[j][1][node]=u                                             */
/* UV[j][2][node]=v                                             */

void define_uv_space(E)
     struct All_variables *E;

{

    int kk,j;
    int midnode;
    int numnodes,node;

    double u,v,cosc,theta,phi;
    double theta_f,phi_f;

    if (E->parallel.me==0) fprintf(stderr,"Setting up UV space\n");
    fflush(stderr);

    numnodes=E->lmesh.nno;

    /* open memory for uv space coords */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {

            for (kk=1;kk<=2;kk++)
                {
                    //TODO: allocate for surface nodes only to save memory
                    if ((E->trace.UV[j][kk]=(double *)malloc((numnodes+1)*sizeof(double)))==NULL)
                        {
                            fprintf(E->trace.fpt,"Error(define uv)-not enough memory(a)\n");
                            fflush(E->trace.fpt);
                            exit(10);
                        }
                }

            /* uv space requires a reference point */
            /* UV[j][1][0]=fixed theta */
            /* UV[j][2][0]=fixed phi */


            midnode=numnodes/2;

            E->trace.UV[j][1][0]=E->sx[j][1][midnode];
            E->trace.UV[j][2][0]=E->sx[j][2][midnode];

            theta_f=E->sx[j][1][midnode];
            phi_f=E->sx[j][2][midnode];

            E->trace.cos_theta_f=cos(theta_f);
            E->trace.sin_theta_f=sin(theta_f);

            /* convert each nodal point to u and v */

            for (node=1;node<=numnodes;node++)
                {
                    theta=E->sx[j][1][node];
                    phi=E->sx[j][2][node];

                    cosc=cos(theta_f)*cos(theta)+sin(theta_f)*sin(theta)*
                        cos(phi-phi_f);
                    u=sin(theta)*sin(phi-phi_f)/cosc;
                    v=(sin(theta_f)*cos(theta)-cos(theta_f)*sin(theta)*cos(phi-phi_f))/cosc;

                    E->trace.UV[j][1][node]=u;
                    E->trace.UV[j][2][node]=v;

                }


        }/*end cap */

    return;
}

/***************************************************************/
/* DETERMINE SHAPE COEFFICIENTS                                */
/*                                                             */
/* An initialization function that determines the coeffiecients*/
/* to all element shape functions.                             */
/* This method uses standard linear shape functions of         */
/* triangular elements. (See Cuvelier, Segal, and              */
/* van Steenhoven, 1986). This is all in UV space.             */
/*                                                             */
/* shape_coefs[cap][wedge][3 shape functions*3 coefs][nelems]  */

void determine_shape_coefficients(E)
     struct All_variables *E;
{

    int j,nelem,iwedge,kk;
    int node;

    double u[5],v[5];
    double x1=0.0;
    double x2=0.0;
    double x3=0.0;
    double y1=0.0;
    double y2=0.0;
    double y3=0.0;
    double delta,a0,a1,a2;

    /* really only have to do this for each surface element, but */
    /* for simplicity, it is done for every element              */

    if (E->parallel.me==0) fprintf(stderr," Determining Shape Coefficients\n");
    fflush(stderr);

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {

            /* first, allocate memory */

            for(iwedge=1;iwedge<=2;iwedge++)
                {
                    for (kk=1;kk<=9;kk++)
                        {
                            //TODO: allocate for surface elements only to save memory
                            if ((E->trace.shape_coefs[j][iwedge][kk]=
                                 (double *)malloc((E->lmesh.nel+1)*sizeof(double)))==NULL)
                                {
                                    fprintf(E->trace.fpt,"ERROR(find shape coefs)-not enough memory(a)\n");
                                    fflush(E->trace.fpt);
                                    exit(10);
                                }
                        }
                }

            for (nelem=1;nelem<=E->lmesh.nel;nelem++)
                {

                    /* find u,v of local nodes at one radius  */

                    for(kk=1;kk<=4;kk++)
                        {
                            node=E->IEN[E->mesh.levmax][j][nelem].node[kk];
                            u[kk]=E->trace.UV[j][1][node];
                            v[kk]=E->trace.UV[j][2][node];
                        }

                    for(iwedge=1;iwedge<=2;iwedge++)
                        {


                            if (iwedge==1)
                                {
                                    x1=u[1];
                                    x2=u[2];
                                    x3=u[3];
                                    y1=v[1];
                                    y2=v[2];
                                    y3=v[3];
                                }
                            if (iwedge==2)
                                {
                                    x1=u[1];
                                    x2=u[3];
                                    x3=u[4];
                                    y1=v[1];
                                    y2=v[3];
                                    y3=v[4];
                                }

                            /* shape function 1 */

                            delta=(x3-x2)*(y1-y2)-(y2-y3)*(x2-x1);
                            a0=(x2*y3-x3*y2)/delta;
                            a1=(y2-y3)/delta;
                            a2=(x3-x2)/delta;

                            E->trace.shape_coefs[j][iwedge][1][nelem]=a0;
                            E->trace.shape_coefs[j][iwedge][2][nelem]=a1;
                            E->trace.shape_coefs[j][iwedge][3][nelem]=a2;

                            /* shape function 2 */

                            delta=(x3-x1)*(y2-y1)-(y1-y3)*(x1-x2);
                            a0=(x1*y3-x3*y1)/delta;
                            a1=(y1-y3)/delta;
                            a2=(x3-x1)/delta;

                            E->trace.shape_coefs[j][iwedge][4][nelem]=a0;
                            E->trace.shape_coefs[j][iwedge][5][nelem]=a1;
                            E->trace.shape_coefs[j][iwedge][6][nelem]=a2;

                            /* shape function 3 */

                            delta=(x1-x2)*(y3-y2)-(y2-y1)*(x2-x3);
                            a0=(x2*y1-x1*y2)/delta;
                            a1=(y2-y1)/delta;
                            a2=(x1-x2)/delta;

                            E->trace.shape_coefs[j][iwedge][7][nelem]=a0;
                            E->trace.shape_coefs[j][iwedge][8][nelem]=a1;
                            E->trace.shape_coefs[j][iwedge][9][nelem]=a2;


                        } /* end wedge */
                } /* end elem */
        } /* end cap */


    return;
}

/****************** GET CARTESIAN VELOCITY FIELD **************************************/
/*                                                                                    */
/* This function computes the cartesian velocity field from the spherical             */
/* velocity field computed from main Citcom code.                                     */

void get_cartesian_velocity_field(E)
     struct All_variables *E;
{

    int j,m,i;
    int kk;
    int lev = E->mesh.levmax;

    double sint,sinf,cost,cosf;
    double v_theta,v_phi,v_rad;
    double vx,vy,vz;

    static int been_here=0;

    if (been_here==0)
        {
            for (j=1;j<=E->sphere.caps_per_proc;j++)
                {
                    for (kk=1;kk<=3;kk++)
                        {
                            if ((E->trace.V0_cart[j][kk]=(double *)malloc((E->lmesh.nno+1)*sizeof(double)))==NULL)
                                {
                                    fprintf(E->trace.fpt,"ERROR(get_cartesian_velocity)-no memory 82hd7\n");
                                    fflush(E->trace.fpt);
                                    exit(10);
                                }
                        }
                }

            been_here++;
        }


    for (m=1;m<=E->sphere.caps_per_proc;m++)
        {

            for (i=1;i<=E->lmesh.nno;i++)
                {
                    sint=E->SinCos[lev][m][0][i];
                    sinf=E->SinCos[lev][m][1][i];
                    cost=E->SinCos[lev][m][2][i];
                    cosf=E->SinCos[lev][m][3][i];

                    v_theta=E->sphere.cap[m].V[1][i];
                    v_phi=E->sphere.cap[m].V[2][i];
                    v_rad=E->sphere.cap[m].V[3][i];



                    vx=v_theta*cost*cosf-v_phi*sinf+v_rad*sint*cosf;
                    vy=v_theta*cost*sinf+v_phi*cosf+v_rad*sint*sinf;
                    vz=-v_theta*sint+v_rad*cost;

                    E->trace.V0_cart[m][1][i]=vx;
                    E->trace.V0_cart[m][2][i]=vy;
                    E->trace.V0_cart[m][3][i]=vz;
                }
        }

    return;
}

/*********** KEEP IN SPHERE *********************************************/
/*                                                                      */
/* This function makes sure the particle is within the sphere, and      */
/* phi and theta are within the proper degree range.                    */

void keep_in_sphere(E,x,y,z,theta,phi,rad)
     struct All_variables *E;
     double *x;
     double *y;
     double *z;
     double *theta;
     double *phi;
     double *rad;
{
    fix_theta_phi(theta, phi);
    fix_radius(E,rad,theta,phi,x,y,z);

    return;
}

/*********** CHECK SUM **************************************************/
/*                                                                      */
/* This functions checks to make sure number of tracers is preserved    */

void check_sum(E)
     struct All_variables *E;
{

    int number,iold_number;

    number=isum_tracers(E);

    iold_number=E->trace.ilast_tracer_count;

    if (number!=iold_number)
        {
            fprintf(E->trace.fpt,"ERROR(check_sum)-break in conservation %d %d\n",
                    number,iold_number);
            fflush(E->trace.fpt);
            exit(10);
        }

    E->trace.ilast_tracer_count=number;

    return;
}

/************* ISUM TRACERS **********************************************/
/*                                                                       */
/* This function uses MPI to sum all tracers and returns number of them. */

static int isum_tracers(struct All_variables *E)
{


    int imycount;
    int iallcount;
    int num;
    int j;

    iallcount=0;

    imycount=0;
    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            imycount=imycount+E->trace.ntracers[j];
        }

    MPI_Allreduce(&imycount,&iallcount,1,MPI_INT,MPI_SUM,E->parallel.world);

    num=iallcount;

    return num;
}



/* &&&&&&&&&&&&&&&&&&&& ANALYTICAL TESTS &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&**************/

/**************** ANALYTICAL TEST *********************************************************/
/*                                                                                        */
/* This function (and the 2 following) are used to test advection of tracers by assigning */
/* a test function (in "analytical_test_function").                                       */

void analytical_test(E)
     struct All_variables *E;

{

    int kk,pp;
    int nsteps;
    int j;
    int my_number,number;
    int nrunge_steps;
    int nrunge_refinement;

    double dt;
    double runge_dt;
    double theta,phi,rad;
    double time;
    double vel_s[4];
    double vel_c[4];
    double my_theta0,my_phi0,my_rad0;
    double my_thetaf,my_phif,my_radf;
    double theta0,phi0,rad0;
    double thetaf,phif,radf;
    double x0_s[4],xf_s[4];
    double x0_c[4],xf_c[4];
    double vec[4];
    double runge_path_length,runge_time;
    double x0,y0,z0;
    double xf,yf,zf;
    double difference;
    double difperpath;

    void analytical_test_function();
    void predict_tracers();
    void correct_tracers();
    void analytical_runge_kutte();
    void sphere_to_cart();


    fprintf(E->trace.fpt,"Starting Analytical Test\n");
    if (E->parallel.me==0) fprintf(stderr,"Starting Analytical Test\n");
    fflush(E->trace.fpt);
    fflush(stderr);

    /* Reset Box cushion to 0 */

    E->trace.box_cushion=0.0000;

    /* test paramters */

    nsteps=200;
    dt=0.0001;

    E->advection.timestep=dt;

    fprintf(E->trace.fpt,"steps: %d  dt: %f\n",nsteps,dt);

    /* Assign test velocity to Citcom nodes */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            for (kk=1;kk<=E->lmesh.nno;kk++)
                {

                    theta=E->sx[j][1][kk];
                    phi=E->sx[j][2][kk];
                    rad=E->sx[j][3][kk];

                    analytical_test_function(E,theta,phi,rad,vel_s,vel_c);

                    E->sphere.cap[j].V[1][kk]=vel_s[1];
                    E->sphere.cap[j].V[2][kk]=vel_s[2];
                    E->sphere.cap[j].V[3][kk]=vel_s[3];
                }
        }

    time=0.0;

    my_theta0=0.0;
    my_phi0=0.0;
    my_rad0=0.0;
    my_thetaf=0.0;
    my_phif=0.0;
    my_radf=0.0;

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            if (E->trace.ntracers[j]>10)
                {
                    fprintf(E->trace.fpt,"Warning(analytical)-too many tracers to print!\n");
                    fflush(E->trace.fpt);
                    if (E->trace.itracer_warnings==1) exit(10);
                }
        }

    /* print initial positions */

    E->monitor.solution_cycles=0;
    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            for (pp=1;pp<=E->trace.ntracers[j];pp++)
                {
                    theta=E->trace.basicq[j][0][pp];
                    phi=E->trace.basicq[j][1][pp];
                    rad=E->trace.basicq[j][2][pp];

                    fprintf(E->trace.fpt,"(%d) time: %f theta: %f phi: %f rad: %f\n",E->monitor.solution_cycles,time,theta,phi,rad);

                    if (pp==1) fprintf(stderr,"(%d) time: %f theta: %f phi: %f rad: %f\n",E->monitor.solution_cycles,time,theta,phi,rad);

                    if (pp==1)
                        {
                            my_theta0=theta;
                            my_phi0=phi;
                            my_rad0=rad;
                        }
                }
        }

    /* advect tracers */

    for (kk=1;kk<=nsteps;kk++)
        {
            E->monitor.solution_cycles=kk;

            time=time+dt;

            predict_tracers(E);
            correct_tracers(E);

            for (j=1;j<=E->sphere.caps_per_proc;j++)
                {
                    for (pp=1;pp<=E->trace.ntracers[j];pp++)
                        {
                            theta=E->trace.basicq[j][0][pp];
                            phi=E->trace.basicq[j][1][pp];
                            rad=E->trace.basicq[j][2][pp];

                            fprintf(E->trace.fpt,"(%d) time: %f theta: %f phi: %f rad: %f\n",E->monitor.solution_cycles,time,theta,phi,rad);

                            if (pp==1) fprintf(stderr,"(%d) time: %f theta: %f phi: %f rad: %f\n",E->monitor.solution_cycles,time,theta,phi,rad);

                            if ((kk==nsteps) && (pp==1))
                                {
                                    my_thetaf=theta;
                                    my_phif=phi;
                                    my_radf=rad;
                                }
                        }
                }

        }

    /* Get ready for comparison to Runge-Kutte (only works for one tracer) */

    fflush(E->trace.fpt);
    fflush(stderr);
    parallel_process_sync(E);

    fprintf(E->trace.fpt,"\n\nComparison to Runge-Kutte\n");
    if (E->parallel.me==0) fprintf(stderr,"Comparison to Runge-Kutte\n");

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            my_number=E->trace.ntracers[j];
        }

    MPI_Allreduce(&my_number,&number,1,MPI_INT,MPI_SUM,E->parallel.world);

    fprintf(E->trace.fpt,"Number of tracers: %d\n", number);
    if (E->parallel.me==0) fprintf(stderr,"Number of tracers: %d\n", number);

    /* if more than 1 tracer, exit */

    if (number!=1)
        {
            fprintf(E->trace.fpt,"(Note: RK comparison only appropriate for one tracing particle (%d here) \n",number);
            if (E->parallel.me==0) fprintf(stderr,"(Note: RK comparison only appropriate for one tracing particle (%d here) \n",number);
            fflush(E->trace.fpt);
            fflush(stderr);
            parallel_process_termination();
        }


    /* communicate starting and final positions */

    MPI_Allreduce(&my_theta0,&theta0,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    MPI_Allreduce(&my_phi0,&phi0,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    MPI_Allreduce(&my_rad0,&rad0,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    MPI_Allreduce(&my_thetaf,&thetaf,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    MPI_Allreduce(&my_phif,&phif,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    MPI_Allreduce(&my_radf,&radf,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);

    x0_s[1]=theta0;
    x0_s[2]=phi0;
    x0_s[3]=rad0;

    nrunge_refinement=1000;

    nrunge_steps=nsteps*nrunge_refinement;
    runge_dt=dt/(1.0*nrunge_refinement);


    analytical_runge_kutte(E,nrunge_steps,runge_dt,x0_s,x0_c,xf_s,xf_c,vec);

    runge_time=vec[1];
    runge_path_length=vec[2];

    /* initial coordinates - both citcom and RK */

    x0=x0_c[1];
    y0=x0_c[2];
    z0=x0_c[3];

    /* convert final citcom coords into cartesian */

    sphere_to_cart(E,thetaf,phif,radf,&xf,&yf,&zf);

    difference=sqrt((xf-xf_c[1])*(xf-xf_c[1])+(yf-xf_c[2])*(yf-xf_c[2])+(zf-xf_c[3])*(zf-xf_c[3]));

    difperpath=difference/runge_path_length;

    /* Print out results */

    fprintf(E->trace.fpt,"Citcom calculation: steps: %d  dt: %f\n",nsteps,dt);
    fprintf(E->trace.fpt,"  (nodes per cap: %d x %d x %d)\n",E->lmesh.nox,E->lmesh.noy,(E->lmesh.noz-1)*E->parallel.nprocz+1);
    fprintf(E->trace.fpt,"                    starting position: theta: %f phi: %f rad: %f\n", theta0,phi0,rad0);
    fprintf(E->trace.fpt,"                    final position: theta: %f phi: %f rad: %f\n", thetaf,phif,radf);
    fprintf(E->trace.fpt,"                    (final time: %f) \n",time );

    fprintf(E->trace.fpt,"\n\nRunge-Kutte calculation: steps: %d  dt: %g\n",nrunge_steps,runge_dt);
    fprintf(E->trace.fpt,"                    starting position: theta: %f phi: %f rad: %f\n", theta0,phi0,rad0);
    fprintf(E->trace.fpt,"                    final position: theta: %f phi: %f rad: %f\n",xf_s[1],xf_s[2],xf_s[3]);
    fprintf(E->trace.fpt,"                    path length: %f \n",runge_path_length );
    fprintf(E->trace.fpt,"                    (final time: %f) \n",runge_time );

    fprintf(E->trace.fpt,"\n\n Difference between Citcom and RK: %e  (diff per path length: %e)\n\n",difference,difperpath);

    if (E->parallel.me==0)
        {
            fprintf(stderr,"Citcom calculation: steps: %d  dt: %f\n",nsteps,dt);
            fprintf(stderr,"  (nodes per cap: %d x %d x %d)\n",E->lmesh.nox,E->lmesh.noy,(E->lmesh.noz-1)*E->parallel.nprocz+1);
            fprintf(stderr,"                    starting position: theta: %f phi: %f rad: %f\n", theta0,phi0,rad0);
            fprintf(stderr,"                    final position: theta: %f phi: %f rad: %f\n", thetaf,phif,radf);
            fprintf(stderr,"                    (final time: %f) \n",time );

            fprintf(stderr,"\n\nRunge-Kutte calculation: steps: %d  dt: %f\n",nrunge_steps,runge_dt);
            fprintf(stderr,"                    starting position: theta: %f phi: %f rad: %f\n", theta0,phi0,rad0);
            fprintf(stderr,"                    final position: theta: %f phi: %f rad: %f\n",xf_s[1],xf_s[2],xf_s[3]);
            fprintf(stderr,"                    path length: %f \n",runge_path_length );
            fprintf(stderr,"                    (final time: %f) \n",runge_time );

            fprintf(stderr,"\n\n Difference between Citcom and RK: %e  (diff per path length: %e)\n\n",difference,difperpath);

        }

    fflush(E->trace.fpt);
    fflush(stderr);

    return;
}

/*************** ANALYTICAL RUNGE KUTTE ******************/
/*                                                       */
void analytical_runge_kutte(E,nsteps,dt,x0_s,x0_c,xf_s,xf_c,vec)
     struct All_variables *E;
     int nsteps;
     double dt;
     double *x0_c;
     double *x0_s;
     double *xf_c;
     double *xf_s;
     double *vec;

{

    int kk;

    double x_0,y_0,z_0;
    double x_p,y_p,z_p;
    double x_c=0.0;
    double y_c=0.0;
    double z_c=0.0;
    double theta_0,phi_0,rad_0;
    double theta_p,phi_p,rad_p;
    double theta_c,phi_c,rad_c;
    double vel0_s[4],vel0_c[4];
    double velp_s[4],velp_c[4];
    double time;
    double path,dtpath;

    void sphere_to_cart();
    void cart_to_sphere();
    void analytical_test_function();

    theta_0=x0_s[1];
    phi_0=x0_s[2];
    rad_0=x0_s[3];

    sphere_to_cart(E,theta_0,phi_0,rad_0,&x_0,&y_0,&z_0);

    /* fill initial cartesian vector to send back */

    x0_c[1]=x_0;
    x0_c[2]=y_0;
    x0_c[3]=z_0;

    time=0.0;
    path=0.0;

    for (kk=1;kk<=nsteps;kk++)
        {

            /* get velocity at initial position */

            analytical_test_function(E,theta_0,phi_0,rad_0,vel0_s,vel0_c);

            /* Find predicted midpoint position */

            x_p=x_0+vel0_c[1]*dt*0.5;
            y_p=y_0+vel0_c[2]*dt*0.5;
            z_p=z_0+vel0_c[3]*dt*0.5;

            /* convert to spherical */

            cart_to_sphere(E,x_p,y_p,z_p,&theta_p,&phi_p,&rad_p);

            /* get velocity at predicted midpoint position */

            analytical_test_function(E,theta_p,phi_p,rad_p,velp_s,velp_c);

            /* Find corrected position using midpoint velocity */

            x_c=x_0+velp_c[1]*dt;
            y_c=y_0+velp_c[2]*dt;
            z_c=z_0+velp_c[3]*dt;

            /* convert to spherical */

            cart_to_sphere(E,x_c,y_c,z_c,&theta_c,&phi_c,&rad_c);

            /* compute path lenght */

            dtpath=sqrt((x_c-x_0)*(x_c-x_0)+(y_c-y_0)*(y_c-y_0)+(z_c-z_0)*(z_c-z_0));
            path=path+dtpath;

            time=time+dt;

            x_0=x_c;
            y_0=y_c;
            z_0=z_c;

            /* next time step */

        }

    /* fill final spherical and cartesian vectors to send back */

    xf_s[1]=theta_c;
    xf_s[2]=phi_c;
    xf_s[3]=rad_c;

    xf_c[1]=x_c;
    xf_c[2]=y_c;
    xf_c[3]=z_c;

    vec[1]=time;
    vec[2]=path;

    return;
}



/********************************************************************/
/* This function computes the number of tracers in each element.    */
/* Each tracer can be of different "type", which is the 0th index   */
/* of extraq. How to interprete "type" is left for the application. */

void accumulate_tracers_in_element(struct All_variables *E)
{
    /* how many types of tracers? */
    // TODO: fix to 1, generalized it later
    const int itypes = 1;
    int kk;
    int numtracers;
    int nelem;
    int j;

    for (j=1;j<=E->sphere.caps_per_proc;j++) {

        /* first zero arrays */
        for (kk=1;kk<=E->lmesh.nel;kk++) {
        }

        numtracers=E->trace.ntracers[j];

        /* Fill arrays */
        for (kk=1;kk<=numtracers;kk++) {

            nelem=E->trace.ielement[j][kk];

            //E->composition.itypes[j][nelem]++;


        }
    }

    return;
}


/**************** ANALYTICAL TEST FUNCTION ******************/
/*                                                          */
/* vel_s[1] => velocity in theta direction                  */
/* vel_s[2] => velocity in phi direction                    */
/* vel_s[3] => velocity in radial direction                 */
/*                                                          */
/* vel_c[1] => velocity in x direction                      */
/* vel_c[2] => velocity in y direction                      */
/* vel_c[3] => velocity in z direction                      */

void analytical_test_function(E,theta,phi,rad,vel_s,vel_c)
     struct All_variables *E;
     double theta,phi,rad;
     double *vel_s;
     double *vel_c;

{

    double sint,sinf,cost,cosf;
    double v_theta,v_phi,v_rad;
    double vx,vy,vz;

    /* This is where the function is given in spherical */

    v_theta=50.0*rad*cos(phi);
    v_phi=100.0*rad*sin(theta);
    v_rad=25.0*rad;

    vel_s[1]=v_theta;
    vel_s[2]=v_phi;
    vel_s[3]=v_rad;

    /* Convert the function into cartesian */

    sint=sin(theta);
    sinf=sin(phi);
    cost=cos(theta);
    cosf=cos(phi);

    vx=v_theta*cost*cosf-v_phi*sinf+v_rad*sint*cosf;
    vy=v_theta*cost*sinf+v_phi*cosf+v_rad*sint*sinf;
    vz=-v_theta*sint+v_rad*cost;

    vel_c[1]=vx;
    vel_c[2]=vy;
    vel_c[3]=vz;

    return;
}


/******************* get_neighboring_caps ************************************/
/*                                                                           */
/* Communicate with neighboring processors to get their cap boundaries,      */
/* which is later used by icheck_cap()                                       */
/*                                                                           */

static void get_neighboring_caps(struct All_variables *E)
{
    void sphere_to_cart();

    const int ncorners = 4; /* # of top corner nodes */
    int i, j, n, d, kk, lev, idb;
    int num_ngb, neighbor_proc, tag;
    MPI_Status status[200];
    MPI_Request request[200];

    int node[ncorners];
    double xx[ncorners*3], rr[12][ncorners*3];
    int nox,noy,noz,dims;
    double x,y,z;
    double theta,phi,rad;

    dims=E->mesh.nsd;
    nox=E->lmesh.nox;
    noy=E->lmesh.noy;
    noz=E->lmesh.noz;

    node[0]=nox*noz*(noy-1)+noz;
    node[1]=noz;
    node[2]=noz*nox;
    node[3]=noz*nox*noy;

    lev = E->mesh.levmax;
    tag = 45;

    for (j=1; j<=E->sphere.caps_per_proc; j++) {

        /* loop over top corners to get their coordinates */
        n = 0;
        for (i=0; i<ncorners; i++) {
            for (d=0; d<dims; d++) {
                xx[n] = E->sx[j][d+1][node[i]];
                n++;
            }
        }

        idb = 0;
        num_ngb = E->parallel.TNUM_PASS[lev][j];
        for (kk=1; kk<=num_ngb; kk++) {
            neighbor_proc = E->parallel.PROCESSOR[lev][j].pass[kk];

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
            for (i=1; i<=ncorners; i++) {
                theta = rr[kk][(i-1)*dims];
                phi = rr[kk][(i-1)*dims+1];
                rad = rr[kk][(i-1)*dims+2];

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

        /* debugging output */
        for (kk=0; kk<=num_ngb; kk++) {
            for (i=1; i<=ncorners; i++) {
                fprintf(E->trace.fpt, "pass=%d corner=%d sx=(%g, %g, %g)\n",
                        kk, i,
                        E->trace.theta_cap[kk][i],
                        E->trace.phi_cap[kk][i],
                        E->trace.rad_cap[kk][i]);
            }
        }
        fflush(E->trace.fpt);

    }

    return;
}


/**** PDEBUG ***********************************************************/
static void pdebug(struct All_variables *E, int i)
{

    fprintf(E->trace.fpt,"HERE (Before Sync): %d\n",i);
    fflush(E->trace.fpt);
    parallel_process_sync(E);
    fprintf(E->trace.fpt,"HERE (After Sync): %d\n",i);
    fflush(E->trace.fpt);

    return;
}
