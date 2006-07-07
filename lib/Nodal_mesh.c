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
/* Functions relating to the building and use of mesh locations ... */


#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"

extern int Emergency_stop;

/*
void flogical_mesh_to_real(E,data,level)
     struct All_variables *E;
     float *data;
     int level;

{ int i,j,n1,n2;

  return;
}
*/

void p_to_nodes(E,P,PN,lev)
     struct All_variables *E;
     double **P;
     float **PN;
     int lev;

{ int e,element,node,j,m;

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(node=1;node<=E->lmesh.NNO[lev];node++)
      PN[m][node] =  0.0;
	  
  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(element=1;element<=E->lmesh.NEL[lev];element++)
       for(j=1;j<=enodes[E->mesh.nsd];j++)  {
     	  node = E->IEN[lev][m][element].node[j];
    	  PN[m][node] += P[m][element] * E->TWW[lev][m][element].node[j] ; 
    	  }
 
   (E->exchange_node_f)(E,PN,lev);

   for(m=1;m<=E->sphere.caps_per_proc;m++)
     for(node=1;node<=E->lmesh.NNO[lev];node++)
        PN[m][node] *= E->MASS[lev][m][node];

     return; 
}


/*
void p_to_centres(E,PN,P,lev)
     struct All_variables *E;
     float **PN;
     double **P;
     int lev;

{  int p,element,node,j,m;
   double weight;

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(p=1;p<=E->lmesh.NEL[lev];p++)
      P[m][p] = 0.0;

   weight=1.0/((double)enodes[E->mesh.nsd]) ;
   
  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(p=1;p<=E->lmesh.NEL[lev];p++)
      for(j=1;j<=enodes[E->mesh.nsd];j++)
        P[m][p] += PN[m][E->IEN[lev][m][p].node[j]] * weight;

   return;  
   }
*/

/*
void v_to_intpts(E,VN,VE,lev)
  struct All_variables *E;
  float **VN,**VE;
  int lev;
  {

   int m,e,i,j,k;
   const int nsd=E->mesh.nsd;
   const int vpts=vpoints[nsd];
   const int ends=enodes[nsd];

 for (m=1;m<=E->sphere.caps_per_proc;m++)
   for(e=1;e<=E->lmesh.NEL[lev];e++)
     for(i=1;i<=vpts;i++)                 {
        VE[m][(e-1)*vpts + i] = 0.0;
        for(j=1;j<=ends;j++)
          VE[m][(e-1)*vpts + i] += VN[m][E->IEN[lev][m][e].node[j]]*E->N.vpt[GNVINDEX(j,i)];
        }

   return;
  }
*/

/*
void visc_to_intpts(E,VN,VE,lev)
   struct All_variables *E;
   float **VN,**VE;
   int lev;
   {

   int m,e,i,j,k;
   const int nsd=E->mesh.nsd;
   const int vpts=vpoints[nsd];
   const int ends=enodes[nsd];

 for (m=1;m<=E->sphere.caps_per_proc;m++)
   for(e=1;e<=E->lmesh.NEL[lev];e++)
     for(i=1;i<=vpts;i++) {
        VE[m][(e-1)*vpts + i] = 0.0;
	for(j=1;j<=ends;j++)
          VE[m][(e-1)*vpts + i] += log(VN[m][E->IEN[lev][m][e].node[j]]) *  E->N.vpt[GNVINDEX(j,i)];
        VE[m][(e-1)*vpts + i] = exp(VE[m][(e-1)*vpts + i]);
        }

  }
*/

void visc_from_gint_to_nodes(E,VE,VN,lev)
  struct All_variables *E;
  float **VE,**VN;
  int lev;
  {
  int m,e,i,j,k,n;
  const int nsd=E->mesh.nsd;
  const int vpts=vpoints[nsd];
  const int ends=enodes[nsd];
  double temp_visc;

 for (m=1;m<=E->sphere.caps_per_proc;m++)
   for(i=1;i<=E->lmesh.NNO[lev];i++)
     VN[m][i] = 0.0;

 for (m=1;m<=E->sphere.caps_per_proc;m++)
   for(e=1;e<=E->lmesh.NEL[lev];e++)   {
     temp_visc=0.0;
     for(i=1;i<=vpts;i++)
        temp_visc += VE[m][(e-1)*vpts + i];
     temp_visc = temp_visc/vpts;

     for(j=1;j<=ends;j++)                {
       n = E->IEN[lev][m][e].node[j];
       VN[m][n] += E->TWW[lev][m][e].node[j] * temp_visc;
       }
    }
 
   (E->exchange_node_f)(E,VN,lev);

   for(m=1;m<=E->sphere.caps_per_proc;m++)
     for(n=1;n<=E->lmesh.NNO[lev];n++) 
        VN[m][n] *= E->MASS[lev][m][n];

   return;
}


void visc_from_nodes_to_gint(E,VN,VE,lev)
  struct All_variables *E;
  float **VE,**VN;
  int lev;
  {

  int m,e,i,j,k,n;
  const int nsd=E->mesh.nsd;
  const int vpts=vpoints[nsd];
  const int ends=enodes[nsd];
  double temp_visc;

 for (m=1;m<=E->sphere.caps_per_proc;m++)
   for(e=1;e<=E->lmesh.NEL[lev];e++)
     for(i=1;i<=vpts;i++)
       VE[m][(e-1)*vpts+i] = 0.0;

 for (m=1;m<=E->sphere.caps_per_proc;m++)
   for(e=1;e<=E->lmesh.NEL[lev];e++)
     for(i=1;i<=vpts;i++)      {
       temp_visc=0.0;
       for(j=1;j<=ends;j++)             
	 temp_visc += E->N.vpt[GNVINDEX(j,i)]*VN[m][E->IEN[lev][m][e].node[j]];

       VE[m][(e-1)*vpts+i] = temp_visc;
       }

   return;
   }

void visc_from_gint_to_ele(E,VE,VN,lev)
  struct All_variables *E;
  float **VE,**VN;
  int lev;
  {
  int m,e,i,j,k,n;
  const int nsd=E->mesh.nsd;
  const int vpts=vpoints[nsd];
  const int ends=enodes[nsd];
  double temp_visc;

 for (m=1;m<=E->sphere.caps_per_proc;m++)
   for(i=1;i<=E->lmesh.NEL[lev];i++)
     VN[m][i] = 0.0;

 for (m=1;m<=E->sphere.caps_per_proc;m++)
   for(e=1;e<=E->lmesh.NEL[lev];e++)   {
     temp_visc=0.0;
     for(i=1;i<=vpts;i++)
        temp_visc += VE[m][(e-1)*vpts + i];
     temp_visc = temp_visc/vpts;

     VN[m][e] = temp_visc;
    }
 
   return;
}


void visc_from_ele_to_gint(E,VN,VE,lev)
  struct All_variables *E;
  float **VE,**VN;
  int lev;
  {

  int m,e,i,j,k,n;
  const int nsd=E->mesh.nsd;
  const int vpts=vpoints[nsd];
  const int ends=enodes[nsd];
  double temp_visc;

 for (m=1;m<=E->sphere.caps_per_proc;m++)
   for(e=1;e<=E->lmesh.NEL[lev];e++)
     for(i=1;i<=vpts;i++)
       VE[m][(e-1)*vpts+i] = 0.0;

 for (m=1;m<=E->sphere.caps_per_proc;m++)
   for(e=1;e<=E->lmesh.NEL[lev];e++)
     for(i=1;i<=vpts;i++)      {

       VE[m][(e-1)*vpts+i] = VN[m][e];
       }

   return;
 }
