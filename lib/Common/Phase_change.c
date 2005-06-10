/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *<LicenseText>
 *=====================================================================
 *
 *                              CitcomS
 *                 ---------------------------------
 *
 *                              Authors:
 *           Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 *           Clint Conrad, Michael Gurnis, and Eun-seo Choi
 *          (c) California Institute of Technology 1994-2005
 *
 *        By downloading and/or installing this software you have
 *       agreed to the CitcomS.py-LICENSE bundled with this software.
 *             Free for non-commercial academic research ONLY.
 *      This program is distributed WITHOUT ANY WARRANTY whatsoever.
 *
 *=====================================================================
 *
 *  Copyright June 2005, by the California Institute of Technology.
 *  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
 * 
 *  Any commercial use must be negotiated with the Office of Technology
 *  Transfer at the California Institute of Technology. This software
 *  may be subject to U.S. export control laws and regulations. By
 *  accepting this software, the user agrees to comply with all
 *  applicable U.S. export laws and regulations, including the
 *  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
 *  the Export Administration Regulations, 15 C.F.R. 730-744. User has
 *  the responsibility to obtain export licenses, or other export
 *  authority as may be required before exporting such information to
 *  foreign countries or providing access to foreign nationals.  In no
 *  event shall the California Institute of Technology be liable to any
 *  party for direct, indirect, special, incidental or consequential
 *  damages, including lost profits, arising out of the use of this
 *  software and its documentation, even if the California Institute of
 *  Technology has been advised of the possibility of such damage.
 * 
 *  The California Institute of Technology specifically disclaims any
 *  warranties, including the implied warranties or merchantability and
 *  fitness for a particular purpose. The software and documentation
 *  provided hereunder is on an "as is" basis, and the California
 *  Institute of Technology has no obligations to provide maintenance,
 *  support, updates, enhancements or modifications.
 *
 *=====================================================================
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
void phase_change_apply(struct All_variables *E, double **buoy,
			float **B, float **B_b,
			float Ra, float clapeyron,
			float depth, float transT, float width);
void calc_phase_change(struct All_variables *E, 
		       float **B, float **B_b,
		       float Ra, float clapeyron,
		       float depth, float transT, float width);
void debug_phase_change(struct All_variables *E, float **B);


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
  if (abs(E->control.Ra_410) > 1e-10) {
    phase_change_apply(E, buoy, E->Fas410, E->Fas410_b, E->control.Ra_410,
		       E->control.clapeyron410, E->viscosity.z410,
		       E->control.transT410, E->control.width410);
				      }
  return;
}


void phase_change_apply_670(struct All_variables *E, double **buoy)
{
  if (abs(E->control.Ra_670) > 1e-10)
    phase_change_apply(E, buoy, E->Fas670, E->Fas670_b, E->control.Ra_670,
		       E->control.clapeyron670, E->viscosity.zlm,
		       E->control.transT670, E->control.width670);
  return;
}


void phase_change_apply_cmb(struct All_variables *E, double **buoy)
{
  if (abs(E->control.Ra_cmb) > 1e-10)
    phase_change_apply(E, buoy, E->Fascmb, E->Fascmb_b, E->control.Ra_cmb,
		       E->control.clapeyroncmb, E->viscosity.zcmb,
		       E->control.transTcmb, E->control.widthcmb);
  return;
}


void phase_change_apply(struct All_variables *E, double **buoy,
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
  }

  return;
}


void calc_phase_change(struct All_variables *E, 
		       float **B, float **B_b,
		       float Ra, float clapeyron,
		       float depth, float transT, float width)
{
  int i,j,k,n,ns,m;
  float e_pressure,pt5,one;

  pt5 = 0.5;
  one = 1.0;

  for(m=1;m<=E->sphere.caps_per_proc;m++)     {
    for(i=1;i<=E->lmesh.nno;i++)  {
      e_pressure = (E->sphere.ro-E->sx[m][3][i]) - depth
	- clapeyron*(E->T[m][i]-transT);
      
      B[m][i] = pt5*(one+tanh(width*e_pressure));
    }

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


void debug_phase_change(struct All_variables *E, float **B)
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



/* The following three functions are obsolete. */

void phase_change_410(E,B,B_b)
  struct All_variables *E;
  float **B,**B_b;
{
  int i,j,k,n,ns,m;
  float e_pressure,pt5,one;

  pt5 = 0.5; one=1.0;

  for(m=1;m<=E->sphere.caps_per_proc;m++)     {
    for(i=1;i<=E->lmesh.nno;i++)  {
      e_pressure = (E->sphere.ro-E->sx[m][3][i])-E->viscosity.z410-
            E->control.clapeyron410*(E->T[m][i]-E->control.transT410);

      B[m][i] = pt5*(one+tanh(E->control.width410*e_pressure));
      }

    ns = 0;
    for (k=1;k<=E->lmesh.noy;k++)
      for (j=1;j<=E->lmesh.nox;j++)  {
        ns = ns + 1;
        B_b[m][ns]=0.0;
        for (i=1;i<E->lmesh.noz;i++)   {
          n = (k-1)*E->lmesh.noz*E->lmesh.nox + (j-1)*E->lmesh.noz + i;
          if (B[m][n]>=pt5&&B[m][n+1]<=pt5)
            B_b[m][ns]=(E->sx[m][3][n+1]-E->sx[m][3][n])*(pt5-B[m][n])/(B[m][n+1]-B[m][n])+E->sx[m][3][n];
          }
        }
    }


  return;
  }


void phase_change_670(E,B,B_b)
  struct All_variables *E;
  float **B,**B_b;
{
  int i,j,k,n,ns,m;
  float e_pressure,pt5,one;

  pt5 = 0.5; one=1.0;

  for(m=1;m<=E->sphere.caps_per_proc;m++)     {
    for(i=1;i<=E->lmesh.nno;i++)  {
      e_pressure = (E->sphere.ro-E->sx[m][3][i])-E->viscosity.zlm-
            E->control.clapeyron670*(E->T[m][i]-E->control.transT670);

      B[m][i] = pt5*(one+tanh(E->control.width670*e_pressure));
      }

    ns = 0;
    for (k=1;k<=E->lmesh.noy;k++)
      for (j=1;j<=E->lmesh.nox;j++)  {
        ns = ns + 1;
        B_b[m][ns]=0.0;
        for (i=1;i<E->lmesh.noz;i++)   {
          n = (k-1)*E->lmesh.noz*E->lmesh.nox + (j-1)*E->lmesh.noz + i;
          if (B[m][n]>=pt5&&B[m][n+1]<=pt5)
            B_b[m][ns]=(E->sx[m][3][n+1]-E->sx[m][3][n])*(pt5-B[m][n])/(B[m][n+1]-B[m][n])+E->sx[m][3][n];
          }
        }
    }


  return;
  }


void phase_change_cmb(E,B,B_b)
  struct All_variables *E;
  float **B,**B_b;
{
  int i,j,k,n,ns,m;
  float e_pressure,pt5,one;

  pt5 = 0.5; one=1.0;

  for(m=1;m<=E->sphere.caps_per_proc;m++)     {
    for(i=1;i<=E->lmesh.nno;i++)  {
      e_pressure = (E->sphere.ro-E->sx[m][3][i])-E->viscosity.zcmb-
            E->control.clapeyroncmb*(E->T[m][i]-E->control.transTcmb);

      B[m][i] = pt5*(one+tanh(E->control.widthcmb*e_pressure));
      }

    ns = 0;
    for (k=1;k<=E->lmesh.noy;k++)
      for (j=1;j<=E->lmesh.nox;j++)  {
        ns = ns + 1;
        B_b[m][ns]=0.0;
        for (i=1;i<E->lmesh.noz;i++)   {
          n = (k-1)*E->lmesh.noz*E->lmesh.nox + (j-1)*E->lmesh.noz + i;
          if (B[m][n]>=pt5&&B[m][n+1]<=pt5)
            B_b[m][ns]=(E->sx[m][3][n+1]-E->sx[m][3][n])*(pt5-B[m][n])/(B[m][n+1]-B[m][n])+E->sx[m][3][n];
          }
        }
    }

  return;
  }
