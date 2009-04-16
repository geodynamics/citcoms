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

#include <math.h>
#include <assert.h>
#include <string.h>

#include "global_defs.h"
#include "lith_age.h"
#include "parsing.h"

void parallel_process_termination();
void temperatures_conform_bcs(struct All_variables *);
double modified_plgndr_a(int, int, double);

#include "initial_temperature.h"
static void debug_tic(struct All_variables *);
static void read_tic_from_file(struct All_variables *);
static void construct_tic_from_input(struct All_variables *);

#ifdef USE_GZDIR
void restart_tic_from_gzdir_file(struct All_variables *);
#endif
#ifdef USE_GGRD
#include "ggrd_handling.h"
#endif


void tic_input(struct All_variables *E)
{

  int m = E->parallel.me;
  int noz = E->lmesh.noz;
  int n;
#ifdef USE_GGRD
  int tmp;
#endif

  input_int("tic_method", &(E->convection.tic_method), "0,0,2", m);

#ifdef USE_GGRD			/* for backward capability */
  input_int("ggrd_tinit", &tmp, "0", m);
  if(tmp){
    E->convection.tic_method = 4; /*  */
    E->control.ggrd.use_temp = 1;
  }
#endif  
  /* When tic_method is 0 (default), the temperature is a linear profile +
     perturbation at some layers.

     When tic_method is -1, the temperature is read in from the
     [datafile_old].velo.[rank].[solution_cycles_init] files.

     When tic_method is 1, the temperature is isothermal (== bottom b.c.) +
     uniformly cold plate (thickness specified by 'half_space_age').

     When tic_method is 2, (tic_method==1) + a hot blob. A user can specify
     the location and radius of the blob, and also the amplitude of temperature
     change in the blob relative to the ambient mantle temperautre
     (E->control.mantle_temp).
        - blob_center: A comma-separated list of three float numbers.
        - blob_radius: A dmensionless length, typically a fraction
                       of the Earth's radius.
        - blob_dT    : Dimensionless temperature.

     When tic_method is 3, the temperature is a linear profile + perturbation
     for whole mantle.

     tic_method is 4: read in initial temperature distribution from a set of netcdf grd
                      files. this required the GGRD extension to be compiled in

  */

    /* This part put a temperature anomaly at depth where the global
       node number is equal to load_depth. The horizontal pattern of
       the anomaly is given by spherical harmonic ll & mm. */

    input_int("num_perturbations", &n, "0,0,PERTURB_MAX_LAYERS", m);

    if (n > 0) {
      E->convection.number_of_perturbations = n;

      if (! input_float_vector("perturbmag", n, E->convection.perturb_mag, m) ) {
	fprintf(stderr,"Missing input parameter: 'perturbmag'\n");
	parallel_process_termination();
      }
      if (! input_int_vector("perturbm", n, E->convection.perturb_mm, m) ) {
	fprintf(stderr,"Missing input parameter: 'perturbm'\n");
	parallel_process_termination();
      }
      if (! input_int_vector("perturbl", n, E->convection.perturb_ll, m) ) {
	fprintf(stderr,"Missing input parameter: 'perturbl'\n");
	parallel_process_termination();
      }
      if (! input_int_vector("perturblayer", n, E->convection.load_depth, m) ) {
	fprintf(stderr,"Missing input parameter: 'perturblayer'\n");
	parallel_process_termination();
      }
    }
    else {
      E->convection.number_of_perturbations = 1;
      E->convection.perturb_mag[0] = 1;
      E->convection.perturb_mm[0] = 2;
      E->convection.perturb_ll[0] = 2;
      E->convection.load_depth[0] = (noz+1)/2;
    }

    input_float("half_space_age", &(E->convection.half_space_age), "40.0,1e-3,nomax", m);
    input_float("mantle_temp",&(E->control.mantle_temp),"1.0",m);


    switch(E->convection.tic_method){
    case 2:			/* blob */
      if( ! input_float_vector("blob_centqer", 3, E->convection.blob_center, m)) {
	assert( E->sphere.caps == 12 || E->sphere.caps == 1 );
	if(E->sphere.caps == 12) { /* Full version: just quit here */
	  fprintf(stderr,"Missing input parameter: 'blob_center'.\n");
	  parallel_process_termination();
	}
	else if(E->sphere.caps == 1) { /* Regional version: put the blob at the center */
	  fprintf(stderr,"Missing input parameter: 'blob_center'. The blob will be placed at the center of the domain.\n");
	  E->convection.blob_center[0] = 0.5*(E->control.theta_min+E->control.theta_max);
	  E->convection.blob_center[1] = 0.5*(E->control.fi_min+E->control.fi_max);
	  E->convection.blob_center[2] = 0.5*(E->sphere.ri+E->sphere.ro);
	}
      }
      input_float("blob_radius", &(E->convection.blob_radius), "0.063,0.0,1.0", m);
      input_float("blob_dT", &(E->convection.blob_dT), "0.18,nomin,nomax", m);
      break;
    case 4:
      /*
	case 4: initial temp from grd files
      */
#ifdef USE_GGRD
      /* 
	 read in some more parameters 
	 
      */
      /* scale the anomalies with PREM densities */
      input_boolean("ggrd_tinit_scale_with_prem",&(E->control.ggrd.temp.scale_with_prem),"off",E->parallel.me);
      /* limit T to 0...1 */
      input_boolean("ggrd_tinit_limit_trange",&(E->control.ggrd.temp.limit_trange),"on",E->parallel.me);
      /* scaling factor for the grids */
      input_double("ggrd_tinit_scale",&(E->control.ggrd.temp.scale),"1.0",E->parallel.me); /* scale */
      /* temperature offset factor */
      input_double("ggrd_tinit_offset",&(E->control.ggrd.temp.offset),"0.0",E->parallel.me); /* offset */
      /* grid name, without the .i.grd suffix */
      input_string("ggrd_tinit_gfile",E->control.ggrd.temp.gfile,"",E->parallel.me); /* grids */
      input_string("ggrd_tinit_dfile",E->control.ggrd.temp.dfile,"",E->parallel.me); /* depth.dat layers of grids*/
      /* override temperature boundary condition? */
      input_boolean("ggrd_tinit_override_tbc",&(E->control.ggrd.temp.override_tbc),"off",E->parallel.me);
      input_string("ggrd_tinit_prem_file",E->control.ggrd.temp.prem.model_filename,"hc/prem/prem.dat", E->parallel.me); /* PREM model filename */
#else
      fprintf(stderr,"tic_method 4 only works for USE_GGRD compiled code\n");
      parallel_process_termination();
#endif
      break;
    } /* no default needed */
    return;
}



void convection_initial_temperature(struct All_variables *E)
{
  void report();

  report(E,"Initialize temperature field");

  if (E->convection.tic_method == -1) {
      /* read temperature from file */
#ifdef USE_GZDIR
      if(strcmp(E->output.format, "ascii-gz") == 0)
          restart_tic_from_gzdir_file(E);
      else
#endif
          read_tic_from_file(E);
  }
  else if (E->control.lith_age)
      lith_age_construct_tic(E);
  else
      construct_tic_from_input(E);

  /* Note: it is the callee's responsibility to conform tbc. */
  /* like a call to temperatures_conform_bcs(E); */

  if (E->control.verbose)
    debug_tic(E);

  return;
}


static void debug_tic(struct All_variables *E)
{
  int m, j;

  fprintf(E->fp_out,"output_temperature\n");
  for(m=1;m<=E->sphere.caps_per_proc;m++)        {
    fprintf(E->fp_out,"for cap %d\n",E->sphere.capid[m]);
    for (j=1;j<=E->lmesh.nno;j++)
      fprintf(E->fp_out,"X = %.6e Z = %.6e Y = %.6e T[%06d] = %.6e \n",E->sx[m][1][j],E->sx[m][2][j],E->sx[m][3][j],j,E->T[m][j]);
  }
  fflush(E->fp_out);

  return;
}



static void read_tic_from_file(struct All_variables *E)
{
  int ii, ll, mm;
  float tt;
  int i, m;
  char output_file[255], input_s[1000];
  FILE *fp;

  float v1, v2, v3, g;

  ii = E->monitor.solution_cycles_init;
  sprintf(output_file,"%s.velo.%d.%d",E->control.old_P_file,E->parallel.me,ii);
  fp=fopen(output_file,"r");
  if (fp == NULL) {
    fprintf(E->fp,"(Initial_temperature.c #1) Cannot open %s\n",output_file);
    parallel_process_termination();
  }

  if (E->parallel.me==0)
    fprintf(E->fp,"Reading %s for initial temperature\n",output_file);

  fgets(input_s,1000,fp);
  sscanf(input_s,"%d %d %f",&ll,&mm,&tt);

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    fgets(input_s,1000,fp);
    sscanf(input_s,"%d %d",&ll,&mm);
    for(i=1;i<=E->lmesh.nno;i++)  {
      fgets(input_s,1000,fp);
      if(sscanf(input_s,"%g %g %g %f",&(v1),&(v2),&(v3),&(g)) != 4) {
        fprintf(stderr,"Error while reading file '%s'\n", output_file);
        exit(8);
      }
      /* Truncate the temperature to be within (0,1). */
      /* This might not be desirable in some situations. */
      E->T[m][i] = max(0.0,min(g,1.0));
    }
  }
  fclose (fp);

  temperatures_conform_bcs(E);

  return;
}


static void linear_temperature_profile(struct All_variables *E)
{
    int m, i, j, k, node;
    int nox, noy, noz;
    double r1;

    nox = E->lmesh.nox;
    noy = E->lmesh.noy;
    noz = E->lmesh.noz;

    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(i=1; i<=noy; i++)
            for(j=1; j<=nox;j ++)
                for(k=1; k<=noz; k++) {
                    node = k + (j-1)*noz + (i-1)*nox*noz;
                    r1 = E->sx[m][3][node];
                    E->T[m][node] = E->control.TBCbotval - (E->control.TBCtopval + E->control.TBCbotval)*(r1 - E->sphere.ri)/(E->sphere.ro - E->sphere.ri);
                }

    return;
}


static void conductive_temperature_profile(struct All_variables *E)
{
    int m, i, j, k, node;
    int nox, noy, noz;
    double r1;

    nox = E->lmesh.nox;
    noy = E->lmesh.noy;
    noz = E->lmesh.noz;

    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(i=1; i<=noy; i++)
            for(j=1; j<=nox;j ++)
                for(k=1; k<=noz; k++) {
                    node = k + (j-1)*noz + (i-1)*nox*noz;
                    r1 = E->sx[m][3][node];
                    E->T[m][node] = (E->control.TBCtopval*E->sphere.ro
                                     - E->control.TBCbotval*E->sphere.ri)
                        / (E->sphere.ro - E->sphere.ri)
                        + (E->control.TBCbotval - E->control.TBCtopval)
                        * E->sphere.ro * E->sphere.ri / r1
                        / (E->sphere.ro - E->sphere.ri);
                }

    return;
}


static void constant_temperature_profile(struct All_variables *E, double mantle_temp)
{
    int m, i;

    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(i=1; i<=E->lmesh.nno; i++)
            E->T[m][i] = mantle_temp;

    return;
}


static void add_top_tbl(struct All_variables *E, double age_in_myrs, double mantle_temp)
{
    int m, i, j, k, node;
    int nox, noy, noz;
    double r1, dT, tmp;

    nox = E->lmesh.nox;
    noy = E->lmesh.noy;
    noz = E->lmesh.noz;

    dT = (mantle_temp - E->control.TBCtopval);
    tmp = 0.5 / sqrt(age_in_myrs / E->data.scalet);

    fprintf(stderr, "%e %e\n", dT, tmp);
    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(i=1; i<=noy; i++)
            for(j=1; j<=nox;j ++)
                for(k=1; k<=noz; k++) {
                    node = k + (j-1)*noz + (i-1)*nox*noz;
                    r1 = E->sx[m][3][node];
                    E->T[m][node] -= dT * erfc(tmp * (E->sphere.ro - r1));
                }

    return;
}


static void add_bottom_tbl(struct All_variables *E, double age_in_myrs, double mantle_temp)
{
    int m, i, j, k, node;
    int nox, noy, noz;
    double r1, dT, tmp;

    nox = E->lmesh.nox;
    noy = E->lmesh.noy;
    noz = E->lmesh.noz;

    dT = (E->control.TBCbotval - mantle_temp);
    tmp = 0.5 / sqrt(age_in_myrs / E->data.scalet);

    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(i=1; i<=noy; i++)
            for(j=1; j<=nox;j ++)
                for(k=1; k<=noz; k++) {
                    node = k + (j-1)*noz + (i-1)*nox*noz;
                    r1 = E->sx[m][3][node];
                    E->T[m][node] += dT * erfc(tmp * (r1 - E->sphere.ri));
                }

    return;
}


static void add_perturbations_at_layers(struct All_variables *E)
{
    /* This function put a temperature anomaly at depth where the global
       node number is equal to load_depth. The horizontal pattern of
       the anomaly is given by wavenumber (in regional model) or
       by spherical harmonic (in global model). */

    int m, i, j, k, node;
    int p, ll, mm, kk;
    int nox, noy, noz, gnoz;
    double t1, f1, tlen, flen, con;

    nox = E->lmesh.nox;
    noy = E->lmesh.noy;
    noz = E->lmesh.noz;
    gnoz = E->mesh.noz;

    for (p=0; p<E->convection.number_of_perturbations; p++) {
        ll = E->convection.perturb_ll[p];
        mm = E->convection.perturb_mm[p];
        kk = E->convection.load_depth[p];
        con = E->convection.perturb_mag[p];

        if ( (kk < 1) || (kk >= gnoz) ) continue; /* layer kk is outside domain */

        k = kk - E->lmesh.nzs + 1; /* convert global nz to local nz */
        if ( (k < 1) || (k >= noz) ) continue; /* layer k is not inside this proc. */
        if (E->parallel.me_loc[1] == 0 && E->parallel.me_loc[2] == 0
            && E->sphere.capid[1] == 1 )
            fprintf(stderr,"Initial temperature perturbation:  layer=%d  mag=%g  l=%d  m=%d\n", kk, con, ll, mm);

        if(E->sphere.caps == 1) {
            /* regional mode, add sinosoidal perturbation */

            tlen = M_PI / (E->control.theta_max - E->control.theta_min);
            flen = M_PI / (E->control.fi_max - E->control.fi_min);

            for(m=1; m<=E->sphere.caps_per_proc; m++)
                for(i=1; i<=noy; i++)
                    for(j=1; j<=nox;j ++) {
                        node = k + (j-1)*noz + (i-1)*nox*noz;
                        t1 = (E->sx[m][1][node] - E->control.theta_min) * tlen;
                        f1 = (E->sx[m][2][node] - E->control.fi_min) * flen;

                        E->T[m][node] += con * cos(ll*t1) * cos(mm*f1);
                    }
        }
        else {
            /* global mode, add spherical harmonics perturbation */

            for(m=1; m<=E->sphere.caps_per_proc; m++)
                for(i=1; i<=noy; i++)
                    for(j=1; j<=nox;j ++) {
                        node = k + (j-1)*noz + (i-1)*nox*noz;
                        t1 = E->sx[m][1][node];
                        f1 = E->sx[m][2][node];

                        E->T[m][node] += con * modified_plgndr_a(ll,mm,t1) * cos(mm*f1);
                    }
        } /* end if */
    } /* end for p */

    return;
}


static void add_perturbations_at_all_layers(struct All_variables *E)
{
    /* This function put a temperature anomaly for whole mantle with
       a sinosoidal amplitude in radial dependence. The horizontal pattern
       of the anomaly is given by wavenumber (in regional model) or
       by spherical harmonic (in global model). */

    int m, i, j, k, node;
    int p, ll, mm;
    int nox, noy, noz, gnoz;
    double r1, t1, f1, tlen, flen, rlen, con;

    nox = E->lmesh.nox;
    noy = E->lmesh.noy;
    noz = E->lmesh.noz;
    gnoz = E->mesh.noz;

    rlen = M_PI / (E->sphere.ro - E->sphere.ri);

    for (p=0; p<E->convection.number_of_perturbations; p++) {
        ll = E->convection.perturb_ll[p];
        mm = E->convection.perturb_mm[p];
        con = E->convection.perturb_mag[p];

        if (E->parallel.me_loc[1] == 0 && E->parallel.me_loc[2] == 0
            && E->sphere.capid[1] == 1 )
            fprintf(stderr,"Initial temperature perturbation:  mag=%g  l=%d  m=%d\n", con, ll, mm);

        if(E->sphere.caps == 1) {
            /* regional mode, add sinosoidal perturbation */

            tlen = M_PI / (E->control.theta_max - E->control.theta_min);
            flen = M_PI / (E->control.fi_max - E->control.fi_min);

            for(m=1; m<=E->sphere.caps_per_proc; m++)
                for(i=1; i<=noy; i++)
                    for(j=1; j<=nox;j ++)
                        for(k=1; k<=noz; k++) {
                            node = k + (j-1)*noz + (i-1)*nox*noz;
                            t1 = (E->sx[m][1][node] - E->control.theta_min) * tlen;
                            f1 = (E->sx[m][2][node] - E->control.fi_min) * flen;
                            r1 = E->sx[m][3][node];

                            E->T[m][node] += con * cos(ll*t1) * cos(mm*f1)
                                * sin((r1-E->sphere.ri) * rlen);
                        }
        }
        else {
            /* global mode, add spherical harmonics perturbation */

            for(m=1; m<=E->sphere.caps_per_proc; m++)
                for(i=1; i<=noy; i++)
                    for(j=1; j<=nox;j ++)
                        for(k=1; k<=noz; k++) {
                            node = k + (j-1)*noz + (i-1)*nox*noz;
                            t1 = E->sx[m][1][node];
                            f1 = E->sx[m][2][node];
                            r1 = E->sx[m][3][node];

                            E->T[m][node] += con * modified_plgndr_a(ll,mm,t1)
                                * (cos(mm*f1) + sin(mm*f1))
                                * sin((r1-E->sphere.ri) * rlen);
                        }
        } /* end if */
    } /* end for p */

    return;
}


static void add_spherical_anomaly(struct All_variables *E)
{
    int i, j ,k , m, node;
    int nox, noy, noz;

    double theta_center, fi_center, r_center;
    double radius, amp;

    double x_center, y_center, z_center;
    double x, y, z, distance;

    noy = E->lmesh.noy;
    nox = E->lmesh.nox;
    noz = E->lmesh.noz;

    theta_center = E->convection.blob_center[0];
    fi_center = E->convection.blob_center[1];
    r_center = E->convection.blob_center[2];
    radius = E->convection.blob_radius;
    amp = E->convection.blob_dT;
    
    fprintf(stderr,"center=(%e %e %e) radius=%e dT=%e\n",
            theta_center, fi_center, r_center, radius, amp);

    x_center = r_center * sin(fi_center) * cos(theta_center);
    y_center = r_center * sin(fi_center) * sin(theta_center);
    z_center = r_center * cos(fi_center);

    /* compute temperature field according to nodal coordinate */
    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(i=1; i<=noy; i++)
            for(j=1; j<=nox;j ++)
                for(k=1; k<=noz; k++) {
                    node = k + (j-1)*noz + (i-1)*nox*noz;

                    x = E->x[m][1][node];
                    y = E->x[m][2][node];
                    z = E->x[m][3][node];

                    distance = sqrt((x-x_center)*(x-x_center) +
                                    (y-y_center)*(y-y_center) +
                                    (z-z_center)*(z-z_center));

                    if (distance < radius)
                        E->T[m][node] += amp * exp(-1.0*distance/radius);
                }
    return;
}


static void construct_tic_from_input(struct All_variables *E)
{
    double mantle_temperature;

    switch (E->convection.tic_method){
    case 0:
        /* a linear temperature profile + perturbations at some layers */
        linear_temperature_profile(E);
        add_perturbations_at_layers(E);
        break;

    case 1:
        /* T=1 for whole mantle +  cold lithosphere TBL */
        mantle_temperature = 1;
        constant_temperature_profile(E, mantle_temperature);
        add_top_tbl(E, E->convection.half_space_age, mantle_temperature);
        break;

    case 2:
        /* T='mantle_temp' for whole mantle + cold lithosphere TBL
           + a spherical anomaly at lower center */
        mantle_temperature = E->control.mantle_temp;
        constant_temperature_profile(E, mantle_temperature);
        add_top_tbl(E, E->convection.half_space_age, mantle_temperature);
        add_spherical_anomaly(E);
        break;

    case 3:
        /* a conductive temperature profile + perturbations at all layers */
        conductive_temperature_profile(E);
        add_perturbations_at_all_layers(E);
        break;

    case 4:
        /* read initial temperature from grd files */
#ifdef USE_GGRD
        if (E->sphere.caps == 1)
            ggrd_reg_temp_init(E);
        else
            ggrd_full_temp_init(E);
#else
        fprintf(stderr,"tic_method 4 only works for USE_GGRD compiled code\n");
        parallel_process_termination();
#endif
        break;
    
    case 10:
        /* T='mantle_temp' for whole mantle + cold lithosphere TBL
           + perturbations at some layers */

        mantle_temperature = E->control.mantle_temp;
        constant_temperature_profile(E, mantle_temperature);
        add_top_tbl(E, E->convection.half_space_age, mantle_temperature);
        add_perturbations_at_all_layers(E);
        break;

    case 11:
        /* T='mantle_temp' for whole mantle + hot CMB TBL
           + perturbations at some layers */

        mantle_temperature = E->control.mantle_temp;
        constant_temperature_profile(E, mantle_temperature);
        add_bottom_tbl(E, E->convection.half_space_age, mantle_temperature);
        add_perturbations_at_all_layers(E);
        break;

    case 12:
        /* T='mantle_temp' for whole mantle + cold lithosphere TBL
           + hot CMB TBL + perturbations at some layers */

        mantle_temperature = E->control.mantle_temp;
        constant_temperature_profile(E, mantle_temperature);
        add_top_tbl(E, E->convection.half_space_age, mantle_temperature);
        add_bottom_tbl(E, E->convection.half_space_age, mantle_temperature);
        add_perturbations_at_all_layers(E);
        break;

    case 90:
        /* for benchmarking purpose */
        /* a constant temperature (0) + single perturbation at mid-layer
           as a delta function in r */

        if((E->parallel.nprocz % 2) == 0) {
            if(E->parallel.me==0)
                fprintf(stderr, "ERROR: tic_method=%d -- nprocz is even, cannot put perturbation on processor boundary!\n",
                        E->convection.tic_method);

            parallel_process_termination();
        }

        constant_temperature_profile(E, 0);

        {
            /* adjust the amplitude of perturbation, so that
             * its integral in r is 1 */
            int mid, k;

            E->convection.number_of_perturbations = 1;

            mid = (E->mesh.noz+1) / 2;
            E->convection.load_depth[0] = mid;

            k = mid - E->lmesh.nzs + 1; /* convert to local nz */
            E->convection.perturb_mag[0] = 0;
            if ( (k > 1) && (k < E->lmesh.noz) ) {
                /* layer k is inside this proc. */
                E->convection.perturb_mag[0] = 2 / (E->sx[1][3][k+1] - E->sx[1][3][k-1]);
            }

        }
        add_perturbations_at_layers(E);
        break;

    case 100:
        /* user-defined initial temperature goes here */
        fprintf(stderr,"Need user definition for initial temperture: 'tic_method=%d'\n",
                E->convection.tic_method);
        parallel_process_termination();
        break;

    default:
        /* unknown option */
        fprintf(stderr,"Invalid value: 'tic_method=%d'\n", E->convection.tic_method);
        parallel_process_termination();
        break;
    }

    temperatures_conform_bcs(E);

    /* debugging the code of expanding spherical harmonics */
    /* debug_sphere_expansion(E);*/
    return;
}


