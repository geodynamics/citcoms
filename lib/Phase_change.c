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
#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include "global_defs.h"

#include "parsing.h"
#include "phase_change.h"

static void phase_change_apply(struct All_variables *E, double **buoy,
			       float **B, float **B_b,
			       float Ra, float clapeyron,
			       float depth, float transT, float inv_width);
static void calc_phase_change(struct All_variables *E,
			      float **B, float **B_b,
			      float Ra, float clapeyron,
			      float depth, float transT, float inv_width);
static void debug_phase_change(struct All_variables *E, float **B);


void phase_change_allocate(struct All_variables *E)
{
  int j;
  int nno  = E->lmesh.nno;
  int nsf  = E->lmesh.nsf;

    E->Fas410[CPPR]   = (float *) malloc((nno+1)*sizeof(float));
    E->Fas410_b[CPPR] = (float *) malloc((nsf+1)*sizeof(float));
    E->Fas670[CPPR]   = (float *) malloc((nno+1)*sizeof(float));
    E->Fas670_b[CPPR] = (float *) malloc((nsf+1)*sizeof(float));
    E->Fascmb[CPPR]   = (float *) malloc((nno+1)*sizeof(float));
    E->Fascmb_b[CPPR] = (float *) malloc((nsf+1)*sizeof(float));
}


void phase_change_input(struct All_variables *E)
{
  int m = E->parallel.me;
  float width;

  /* for phase change 410km  */
  input_float("Ra_410",&(E->control.Ra_410),"0.0",m);
  input_float("clapeyron410",&(E->control.clapeyron410),"0.0",m);
  input_float("transT410",&(E->control.transT410),"0.0",m);
  input_float("width410",&width,"0.0",m);

  if (width!=0.0)
    E->control.inv_width410 = 1.0/width;

  /* for phase change 670km   */
  input_float("Ra_670",&(E->control.Ra_670),"0.0",m);
  input_float("clapeyron670",&(E->control.clapeyron670),"0.0",m);
  input_float("transT670",&(E->control.transT670),"0.0",m);
  input_float("width670",&width,"0.0",m);

  if (width!=0.0)
    E->control.inv_width670 = 1.0/width;

  /* for phase change CMB  */
  input_float("Ra_cmb",&(E->control.Ra_cmb),"0.0",m);
  input_float("clapeyroncmb",&(E->control.clapeyroncmb),"0.0",m);
  input_float("transTcmb",&(E->control.transTcmb),"0.0",m);
  input_float("widthcmb",&width,"0.0",m);

  if (width!=0.0)
    E->control.inv_widthcmb = 1.0/width;


  return;
}


void phase_change_apply_410(struct All_variables *E, double **buoy)
{
  if (E->control.Ra_410 != 0.0)
    phase_change_apply(E, buoy, E->Fas410, E->Fas410_b, E->control.Ra_410,
		       E->control.clapeyron410, E->viscosity.z410,
		       E->control.transT410, E->control.inv_width410);
  return;
}


void phase_change_apply_670(struct All_variables *E, double **buoy)
{
  if (E->control.Ra_670 != 0.0)
    phase_change_apply(E, buoy, E->Fas670, E->Fas670_b, E->control.Ra_670,
		       E->control.clapeyron670, E->viscosity.zlm,
		       E->control.transT670, E->control.inv_width670);
  return;
}


void phase_change_apply_cmb(struct All_variables *E, double **buoy)
{
  if (E->control.Ra_cmb != 0.0)
    phase_change_apply(E, buoy, E->Fascmb, E->Fascmb_b, E->control.Ra_cmb,
		       E->control.clapeyroncmb, E->viscosity.zcmb,
		       E->control.transTcmb, E->control.inv_widthcmb);
  return;
}


static void phase_change_apply(struct All_variables *E, double **buoy,
			       float **B, float **B_b,
			       float Ra, float clapeyron,
			       float depth, float transT, float inv_width)
{
  int m, i;

  calc_phase_change(E, B, B_b, Ra, clapeyron, depth, transT, inv_width);
    for(i=1;i<=E->lmesh.nno;i++)
      buoy[CPPR][i] -= Ra * B[CPPR][i];

  if (E->control.verbose) {
    fprintf(E->fp_out, "Ra=%f, clapeyron=%f, depth=%f, transT=%f, inv_width=%f\n",
	    Ra, clapeyron, depth, transT, inv_width);
    debug_phase_change(E,B);
    fflush(E->fp_out);
  }
}


static void calc_phase_change(struct All_variables *E,
			      float **B, float **B_b,
			      float Ra, float clapeyron,
			      float depth, float transT, float inv_width)
{
  int i,j,k,n,ns,m,nz;
  float e_pressure,pt5,one,dz;

  pt5 = 0.5;
  one = 1.0;

    /* compute phase function B, the concentration of the high pressure
     * phase. B is between 0 and 1. */
    for(i=1;i<=E->lmesh.nno;i++)  {
        nz = ((i-1) % E->lmesh.noz) + 1;
        dz = (E->sphere.ro-E->sx[3][i]) - depth;
        /*XXX: dz*rho[nz]*g[nz] is only a approximation for the reduced
         * pressure, a more accurate formula is:
         *   integral(rho(z)*g(z)*dz) from depth_ph to current depth   */
        e_pressure = dz * E->refstate.rho[nz] * E->refstate.gravity[nz]
            - clapeyron * (E->T[i] - transT);

        B[CPPR][i] = pt5 * (one + tanh(inv_width * e_pressure));
    }

    /* compute the phase boundary, defined as the depth where B==0.5 */
    ns = 0;
    for (k=1;k<=E->lmesh.noy;k++)
      for (j=1;j<=E->lmesh.nox;j++)  {
        ns++;
        B_b[CPPR][ns]=0.0;
        for (i=1;i<E->lmesh.noz;i++)   {
          n = (k-1)*E->lmesh.noz*E->lmesh.nox + (j-1)*E->lmesh.noz + i;
          if (B[CPPR][n]>=pt5 && B[CPPR][n+1]<=pt5)
            B_b[CPPR][ns]=(E->sx[3][n+1]-E->sx[3][n])*(pt5-B[CPPR][n])/(B[CPPR][n+1]-B[CPPR][n])+E->sx[3][n];
	}
      }
}


static void debug_phase_change(struct All_variables *E, float **B)
{
  int m, j;

  fprintf(E->fp_out,"output_phase_change_buoyancy\n");
    fprintf(E->fp_out,"for cap %d\n",E->sphere.capid[CPPR]);
    for (j=1;j<=E->lmesh.nno;j++)
      fprintf(E->fp_out,"Z = %.6e T = %.6e B[%06d] = %.6e \n",E->sx[3][j],E->T[j],j,B[CPPR][j]);
  fflush(E->fp_out);
}
