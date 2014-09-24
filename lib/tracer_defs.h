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



/* forward declaration */
struct All_variables;


struct TRACE{

    FILE *fpt;

    char tracer_file[200];

    int itracer_warnings;
    int ianalytical_tracer_test;
    int ic_method;
    int itperel;
    int itracer_interpolation_scheme;

    double box_cushion;

    /* tracer arrays */
    int number_of_basic_quantities;
    int number_of_extra_quantities;
    int number_of_tracer_quantities;

    double *basicq[100];
    double *extraq[13][100];

    int ntracers[13];
    int max_ntracers[13];
    int *ielement[13];

    int number_of_tracers;

    int ilatersize[13];
    int ilater[13];
    double *rlater[13][100];

    /* tracer flavors */
    int nflavors;
    int **ntracer_flavor[13];

    int ic_method_for_flavors;
    double *z_interface;

    char ggrd_file[255];		/* for grd input */
    int ggrd_layers;



    /* statistical parameters */
    int istat_ichoice[13][5];
    int istat_isend;
    int istat_iempty;
    int istat1;
    int istat_elements_checked;
    int ilast_tracer_count;


    /* timing information */
    double advection_time;
    double find_tracers_time;
    double lost_souls_time;


    /* Mesh information */
    double xcap[13][5];
    double ycap[13][5];
    double zcap[13][5];
    double theta_cap[13][5];
    double phi_cap[13][5];
    double rad_cap[13][5];

    double cos_theta[13][5];
    double sin_theta[13][5];
    double cos_phi[13][5];
    double sin_phi[13][5];



    /*********************/
    /* for globall model */
    /*********************/

    /* regular mesh parameters */
    int numtheta[13];
    int numphi[13];
    unsigned int numregel[13];
    unsigned int numregnodes[13];
    double deltheta[13];
    double delphi[13];
    double thetamax[13];
    double thetamin[13];
    double phimax[13];
    double phimin[13];
    int *regnodetoel[13];
    int *regtoel[13][5];

    /* gnomonic shape functions */
    double *shape_coefs[13][3][10];




    /**********************/
    /* for regional model */
    /**********************/

    double *x_space;
    double *y_space;
    double *z_space;




    /*********************/
    /* function pointers */
    /*********************/

    int (* iget_element)(struct All_variables*, int,
                         double, double, double, double, double, double);

    void (* get_velocity)(struct All_variables*, int,
                          double, double, double, double*);

    void (* keep_within_bounds)(struct All_variables*,
                                double*, double*, double*,
                                double*, double*, double*);
};
