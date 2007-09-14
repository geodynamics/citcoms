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
			       float depth, float transT, float width);
static void calc_phase_change(struct All_variables *E,
			      float **B, float **B_b,
			      float Ra, float clapeyron,
			      float depth, float transT, float width);
static void debug_phase_change(struct All_variables *E, float **B);


void phase_change_allocate(struct All_variables *E)
{
  int j;
  int nno  = E->lmesh.nno;
  int nsf  = E->lmesh.nsf;

  for (j=1;j<=E->sphere.caps_per_proc;j++)  {
    E->Fas410[j]   = (float *) malloc((nno+1)*sizeof(float));
    E->Fas410_b[j] = (float *) malloc((nsf+1)*sizeof(float));
    E->Fas670[j]   = (float *) malloc((nno+1)*sizeof(float));
    E->Fas670_b[j] = (float *) malloc((nsf+1)*sizeof(float));
    E->Fascmb[j]   = (float *) malloc((nno+1)*sizeof(float));
    E->Fascmb_b[j] = (float *) malloc((nsf+1)*sizeof(float));
  }

  return;
}


void phase_change_input(struct All_variables *E)
{
  int m = E->parallel.me;

  /* for phase change 410km  */
  input_float("Ra_410",&(E->control.Ra_410),"0.0",m);
  input_float("clapeyron410",&(E->control.clapeyron410),"0.0",m);
  input_float("transT410",&(E->control.transT410),"0.0",m);
  input_float("width410",&(E->control.width410),"0.0",m);

  if (E->control.width410!=0.0)
    E->control.width410 = 1.0/E->control.width410;

  /* for phase change 670km   */
  input_float("Ra_670",&(E->control.Ra_670),"0.0",m);
  input_float("clapeyron670",&(E->control.clapeyron670),"0.0",m);
  input_float("transT670",&(E->control.transT670),"0.0",m);
  input_float("width670",&(E->control.width670),"0.0",m);

  if (E->control.width670!=0.0)
    E->control.width670 = 1.0/E->control.width670;

  /* for phase change CMB  */
  input_float("Ra_cmb",&(E->control.Ra_cmb),"0.0",m);
  input_float("clapeyroncmb",&(E->control.clapeyroncmb),"0.0",m);
  input_float("transTcmb",&(E->control.transTcmb),"0.0",m);
  input_float("widthcmb",&(E->control.widthcmb),"0.0",m);

  if (E->control.widthcmb!=0.0)
    E->control.widthcmb = 1.0/E->control.widthcmb;


  return;
}


void phase_change_apply_410(struct All_variables *E, double **buoy)
{
  if (fabs(E->control.Ra_410) > 1e-10) {
    phase_change_apply(E, buoy, E->Fas410, E->Fas410_b, E->control.Ra_410,
		       E->control.clapeyron410, E->viscosity.z410,
		       E->control.transT410, E->control.width410);
				      }
  return;
}


void phase_change_apply_670(struct All_variables *E, double **buoy)
{
  if (fabs(E->control.Ra_670) > 1e-10)
    phase_change_apply(E, buoy, E->Fas670, E->Fas670_b, E->control.Ra_670,
		       E->control.clapeyron670, E->viscosity.zlm,
		       E->control.transT670, E->control.width670);
  return;
}


void phase_change_apply_cmb(struct All_variables *E, double **buoy)
{
  if (fabs(E->control.Ra_cmb) > 1e-10)
    phase_change_apply(E, buoy, E->Fascmb, E->Fascmb_b, E->control.Ra_cmb,
		       E->control.clapeyroncmb, E->viscosity.zcmb,
		       E->control.transTcmb, E->control.widthcmb);
  return;
}


static void phase_change_apply(struct All_variables *E, double **buoy,
			       float **B, float **B_b,
			       float Ra, float clapeyron,
			       float depth, float transT, float width)
{
  int m, i;

  calc_phase_change(E, B, B_b, Ra, clapeyron, depth, transT, width);
  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=1;i<=E->lmesh.nno;i++)
      buoy[m][i] -= Ra * B[m][i];

  if (E->control.verbose) {
    fprintf(E->fp_out, "Ra=%f, clapeyron=%f, depth=%f, transT=%f, width=%f\n",
	    Ra, clapeyron, depth, transT, width);
    debug_phase_change(E,B);
    fflush(E->fp_out);
  }

  return;
}


static void calc_phase_change(struct All_variables *E,
			      float **B, float **B_b,
			      float Ra, float clapeyron,
			      float depth, float transT, float width)
{
  int i,j,k,n,ns,m,nz;
  float e_pressure,pt5,one,dz;

  pt5 = 0.5;
  one = 1.0;

  for(m=1;m<=E->sphere.caps_per_proc;m++)     {
    /* compute phase function B, the concentration of the high pressure
     * phase. B is between 0 and 1. */
    for(i=1;i<=E->lmesh.nno;i++)  {
        nz = ((i-1) % E->lmesh.noz) + 1;
        dz = (E->sphere.ro-E->sx[m][3][i]) - depth;
        e_pressure = dz * E->refstate.rho[nz] * E->refstate.gravity[nz]
            - clapeyron * (E->T[m][i] - transT);

        B[m][i] = pt5 * (one + tanh(width * e_pressure));
    }

    /* compute the phase boundary, defined as the depth where B==0.5 */
    ns = 0;
    for (k=1;k<=E->lmesh.noy;k++)
      for (j=1;j<=E->lmesh.nox;j++)  {
        ns++;
        B_b[m][ns]=0.0;
        for (i=1;i<E->lmesh.noz;i++)   {
          n = (k-1)*E->lmesh.noz*E->lmesh.nox + (j-1)*E->lmesh.noz + i;
          if (B[m][n]>=pt5 && B[m][n+1]<=pt5)
            B_b[m][ns]=(E->sx[m][3][n+1]-E->sx[m][3][n])*(pt5-B[m][n])/(B[m][n+1]-B[m][n])+E->sx[m][3][n];
	}
      }
  }

  return;
}


static void debug_phase_change(struct All_variables *E, float **B)
{
  int m, j;

  fprintf(E->fp_out,"output_phase_change_buoyancy\n");
  for(m=1;m<=E->sphere.caps_per_proc;m++)        {
    fprintf(E->fp_out,"for cap %d\n",E->sphere.capid[m]);
    for (j=1;j<=E->lmesh.nno;j++)
      fprintf(E->fp_out,"Z = %.6e T = %.6e B[%06d] = %.6e \n",E->sx[m][3][j],E->T[m][j],j,B[m][j]);
  }
  fflush(E->fp_out);

  return;
}
