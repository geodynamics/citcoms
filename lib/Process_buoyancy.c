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

/*  Here are the routines which process the results of each buoyancy solution, and call
    any relevant output routines. Much of the information has probably been output along
    with the velocity field. (So the velocity vectors and other data are fully in sync).
    However, heat fluxes and temperature averages are calculated here (even when they
    get output the next time around the velocity solver);
    */


#include "element_definitions.h"
#include "global_defs.h"
#include "output.h"
#include <math.h>		/* for sqrt */



static void output_interpolated_fields(struct All_variables *E)
{
    void compute_horiz_avg(struct All_variables *E);
    void full_get_shape_functions(struct All_variables *E,
                                  double shp[9], int nelem,
                                  double theta, double phi, double rad);
    void regional_get_shape_functions(struct All_variables *E,
                                      double shp[9], int nelem,
                                      double theta, double phi, double rad);
    double full_interpolate_data(struct All_variables *E,
                                 double shp[9], double data[9]);
    double regional_interpolate_data(struct All_variables *E,
                                     double shp[9], double data[9]);
    char output_file[256];
    FILE *fp1;
    const int m = 1;
    int n, ncolumns, ncomp;
    double *compositions;

    snprintf(output_file, 255, "%s.intp_fields.%d",
             E->control.data_file, E->parallel.me);
    fp1 = output_open(output_file, "w");

    ncomp = 0;
    compositions = NULL;
    if(E->composition.on) {
        ncomp = E->composition.ncomp;
        compositions = malloc(ncomp * sizeof(double));
        if(compositions == NULL) {
            fprintf(stderr, "output_interpolated_fields(): 2 not enough memory.\n");
            exit(1);
        }
    }

    switch(E->trace.itracer_interpolate_fields) {
    case 1:
    case 2:
    case 3:
        /* Format of the output --
         * 1st line is the header:
         *   [ntracers, model_type, ncolumns, ncompositions]
         * the rest is data:
         *   [flavor0, flavor1, radius, temperature, composition(s)]
         */

        if(E->parallel.me == 0) {
            fprintf(E->fp, "Temperature contrast is %e Kelvin\n",
                    E->data.ref_temperature);
            fprintf(stderr, "Temperature contrast is %e Kelvin\n",
                    E->data.ref_temperature);
        }

        ncolumns = 4;
        if(E->composition.on) {
            ncolumns += E->composition.ncomp;
        }

        /* get the horizontal average of temperature and composition */
        compute_horiz_avg(E);

        fprintf(fp1,"%d %d %d %d\n",
                E->trace.ntracers[m], E->trace.itracer_interpolate_fields,
                ncolumns, ncomp);


        for(n=1; n<=E->trace.ntracers[m]; n++) {
            int i, j, k;
            int nelem, flavor0, flavor1;
            int node[9], nz[9];
            double shpfn[9], data[9];
            double theta, phi, rad;
            double temperature;

            nelem = E->trace.ielement[m][n];
            theta = E->trace.basicq[m][0][n];
            phi = E->trace.basicq[m][1][n];
            rad = E->trace.basicq[m][2][n];

            flavor0 = E->trace.extraq[m][0][n];
            flavor1 = E->trace.extraq[m][1][n];

            /* get shape functions at the tracer location */
            if(E->parallel.nprocxy == 12)
                full_get_shape_functions(E, shpfn, nelem, theta, phi, rad);
            else
                regional_get_shape_functions(E, shpfn, nelem, theta, phi, rad);

            /* fetch element data for interpolation */
            for(i=1; i<=ENODES3D; i++) {
                node[i] = E->ien[m][nelem].node[i];
                nz[i] = (node[i] - 1) % E->lmesh.noz + 1;
            }

            for(i=1; i<=ENODES3D; i++) {
                data[i] = E->T[m][node[i]] - E->Have.T[nz[i]];
            }

            if(E->parallel.nprocxy == 12)
                temperature = full_interpolate_data(E, shpfn, data);
            else
                temperature = regional_interpolate_data(E, shpfn, data);

            /** debug **
            fprintf(E->trace.fpt, "result: %e   data: %e %e %e %e %e %e %e %e\n",
                    temperature, data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
            /**/

            for(j=0; j<E->composition.ncomp; j++) {
                for(i=1; i<=ENODES3D; i++) {
                    data[i] = E->composition.comp_node[m][j][node[i]]
                        - E->Have.C[j][nz[i]];
                }
                if(E->parallel.nprocxy == 12)
                    compositions[j] = full_interpolate_data(E, shpfn, data);
                else
                    compositions[j] = regional_interpolate_data(E, shpfn, data);

                /** debug **
                fprintf(E->trace.fpt, "result: %e   data: %e %e %e %e %e %e %e %e\n",
                        compositions[j], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
                /**/
            }

            /* dimensionalize */
            rad *= 1e3 * E->data.radius_km;
            temperature *= E->data.ref_temperature;

            /* output */
            fprintf(fp1,"%d %d %e %e",
                    flavor0, flavor1, rad, temperature);

            for(j=0; j<E->composition.ncomp; j++) {
                fprintf(fp1," %e", compositions[j]);
            }
            fprintf(fp1, "\n");
        }

        break;
    case 100:
        /* user modification here */
        ncolumns = 2;
        break;
    default:
        if(E->parallel.me == 0) {
            fprintf(stderr, "Paramter `itracer_interpolate_fields' has unknown value: %d", E->trace.itracer_interpolate_fields);
            fprintf(E->fp, "Paramter `itracer_interpolate_fields' has unknown value: %d", E->trace.itracer_interpolate_fields);
        }
        parallel_process_termination();

    }

    if(E->composition.on)
        free(compositions);

    fclose(fp1);
    return;
}


void post_processing(struct All_variables *E)
{
    void dump_and_get_new_tracers_to_interpolate_fields(struct All_variables *E);

    if (E->trace.itracer_interpolate_fields && E->control.tracer) {
        dump_and_get_new_tracers_to_interpolate_fields(E);
        output_interpolated_fields(E);
    }
    return;
}



/* ===================
    Surface heat flux
   =================== */

void heat_flux(E)
    struct All_variables *E;
{
    int m,e,el,i,j,node,lnode;
    float *flux[NCS],*SU[NCS],*RU[NCS];
    float VV[4][9],u[9],T[9],dTdz[9],area,uT;
    float *sum_h;

    void velo_from_element();
    void sum_across_surface();
    void return_horiz_ave();
    void return_horiz_ave_f();

    const int dims=E->mesh.nsd,dofs=E->mesh.dof;
    const int vpts=vpoints[dims];
    const int ppts=ppoints[dims];
    const int ends=enodes[dims];
    const int nno=E->lmesh.nno;
    const int lev = E->mesh.levmax;
    const int sphere_key=1;

  sum_h = (float *) malloc((5)*sizeof(float));
  for(i=0;i<=4;i++)
    sum_h[i] = 0.0;

  for(m=1;m<=E->sphere.caps_per_proc;m++) {

    flux[m] = (float *) malloc((1+nno)*sizeof(float));

    for(i=1;i<=nno;i++)   {
      flux[m][i] = 0.0;
      }

    for(e=1;e<=E->lmesh.nel;e++) {

      velo_from_element(E,VV,m,e,sphere_key);

      for(i=1;i<=vpts;i++)   {
        u[i] = 0.0;
        T[i] = 0.0;
        dTdz[i] = 0.0;
        for(j=1;j<=ends;j++)  {
          u[i] += VV[3][j]*E->N.vpt[GNVINDEX(j,i)];
          T[i] += E->T[m][E->ien[m][e].node[j]]*E->N.vpt[GNVINDEX(j,i)];
          dTdz[i] += -E->T[m][E->ien[m][e].node[j]]*E->gNX[m][e].vpt[GNVXINDEX(2,j,i)];
          }
        }

      uT = 0.0;
      area = 0.0;
      for(i=1;i<=vpts;i++)   {
        /* XXX: missing unit conversion, heat capacity and thermal conductivity */
        uT += u[i]*T[i]*E->gDA[m][e].vpt[i] + dTdz[i]*E->gDA[m][e].vpt[i];
        }

      uT /= E->eco[m][e].area;

      for(j=1;j<=ends;j++)
        flux[m][E->ien[m][e].node[j]] += uT*E->TWW[lev][m][e].node[j];

      }             /* end of e */
    }             /* end of m */


  (E->exchange_node_f)(E,flux,lev);

  for(m=1;m<=E->sphere.caps_per_proc;m++)
     for(i=1;i<=nno;i++)
       flux[m][i] *= E->MASS[lev][m][i];

  if (E->parallel.me_loc[3]==E->parallel.nprocz-1)
    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(i=1;i<=E->lmesh.nsf;i++)
        E->slice.shflux[m][i]=2*flux[m][E->surf_node[m][i]]-flux[m][E->surf_node[m][i]-1];

  if (E->parallel.me_loc[3]==0)
    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(i=1;i<=E->lmesh.nsf;i++)
        E->slice.bhflux[m][i] = 2*flux[m][E->surf_node[m][i]-E->lmesh.noz+1]
                                - flux[m][E->surf_node[m][i]-E->lmesh.noz+2];

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(e=1;e<=E->lmesh.snel;e++) {
         uT =(E->slice.shflux[m][E->sien[m][e].node[1]] +
              E->slice.shflux[m][E->sien[m][e].node[2]] +
              E->slice.shflux[m][E->sien[m][e].node[3]] +
              E->slice.shflux[m][E->sien[m][e].node[4]])*0.25;
         el = e*E->lmesh.elz;
         sum_h[0] += uT*E->eco[m][el].area;
         sum_h[1] += E->eco[m][el].area;

         uT =(E->slice.bhflux[m][E->sien[m][e].node[1]] +
              E->slice.bhflux[m][E->sien[m][e].node[2]] +
              E->slice.bhflux[m][E->sien[m][e].node[3]] +
              E->slice.bhflux[m][E->sien[m][e].node[4]])*0.25;
         el = (e-1)*E->lmesh.elz+1;
         sum_h[2] += uT*E->eco[m][el].area;
         sum_h[3] += E->eco[m][el].area;
         }

  sum_across_surface(E,sum_h,4);

  if (E->parallel.me_loc[3]==E->parallel.nprocz-1)   {
    sum_h[0] = sum_h[0]/sum_h[1];
    /*     if (E->control.verbose && E->parallel.me==E->parallel.nprocz-1) {
	     fprintf(E->fp_out,"surface heat flux= %f %f\n",sum_h[0],E->monitor.elapsed_time);
             fflush(E->fp_out);
    } */
    if (E->parallel.me==E->parallel.nprocz-1) {
      fprintf(stderr,"surface heat flux= %f\n",sum_h[0]);
      //fprintf(E->fp,"surface heat flux= %f\n",sum_h[0]); //commented out because E->fp is only on CPU 0 

      if(E->output.write_q_files > 0){
	/* format: time heat_flow sqrt(v.v)  */
	fprintf(E->output.fpqt,"%13.5e %13.5e %13.5e\n",E->monitor.elapsed_time,sum_h[0],sqrt(E->monitor.vdotv));
	fflush(E->output.fpqt);
      }
    }
  }

  if (E->parallel.me_loc[3]==0)    {
    sum_h[2] = sum_h[2]/sum_h[3];
/*     if (E->control.verbose && E->parallel.me==0) fprintf(E->fp_out,"bottom heat flux= %f %f\n",sum_h[2],E->monitor.elapsed_time); */
    if (E->parallel.me==0) {
      fprintf(stderr,"bottom heat flux= %f\n",sum_h[2]);
      fprintf(E->fp,"bottom heat flux= %f\n",sum_h[2]);
      if(E->output.write_q_files > 0){
	fprintf(E->output.fpqb,"%13.5e %13.5e %13.5e\n",
		E->monitor.elapsed_time,sum_h[2],sqrt(E->monitor.vdotv));
	fflush(E->output.fpqb);
      }

    }
  }


  for(m=1;m<=E->sphere.caps_per_proc;m++)
    free((void *)flux[m]);

  free((void *)sum_h);

  return;
}



/*
  compute horizontal average of temperature, composition and rms velocity
*/
void compute_horiz_avg(struct All_variables *E)
{
    void return_horiz_ave_f();

    int m, n, i;
    float *S1[NCS],*S2[NCS],*S3[NCS];

    for(m=1;m<=E->sphere.caps_per_proc;m++)      {
	S1[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
	S2[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
	S3[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
    }

    for(m=1;m<=E->sphere.caps_per_proc;m++) {
	for(i=1;i<=E->lmesh.nno;i++) {
	    S1[m][i] = E->T[m][i];
	    S2[m][i] = E->sphere.cap[m].V[1][i]*E->sphere.cap[m].V[1][i]
          	+ E->sphere.cap[m].V[2][i]*E->sphere.cap[m].V[2][i];
	    S3[m][i] = E->sphere.cap[m].V[3][i]*E->sphere.cap[m].V[3][i];
	}
    }

    return_horiz_ave_f(E,S1,E->Have.T);
    return_horiz_ave_f(E,S2,E->Have.V[1]);
    return_horiz_ave_f(E,S3,E->Have.V[2]);

    if (E->composition.on) {
        for(n=0; n<E->composition.ncomp; n++) {
            for(m=1;m<=E->sphere.caps_per_proc;m++) {
                for(i=1;i<=E->lmesh.nno;i++)
                    S1[m][i] = E->composition.comp_node[m][n][i];
            }
            return_horiz_ave_f(E,S1,E->Have.C[n]);
        }
    }

    for(m=1;m<=E->sphere.caps_per_proc;m++) {
	free((void *)S1[m]);
	free((void *)S2[m]);
	free((void *)S3[m]);
    }

    for (i=1;i<=E->lmesh.noz;i++) {
	E->Have.V[1][i] = sqrt(E->Have.V[1][i]);
	E->Have.V[2][i] = sqrt(E->Have.V[2][i]);
    }

    return;
}
