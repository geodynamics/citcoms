#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"

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
