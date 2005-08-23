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
  void exchange_node_f();

  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(node=1;node<=E->lmesh.NNO[lev];node++)
      PN[m][node] =  0.0;
	  
  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(element=1;element<=E->lmesh.NEL[lev];element++)
       for(j=1;j<=enodes[E->mesh.nsd];j++)  {
     	  node = E->IEN[lev][m][element].node[j];
    	  PN[m][node] += P[m][element] * E->TWW[lev][m][element].node[j] ; 
    	  }
 
   exchange_node_f (E,PN,lev);

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
  void exchange_node_f();

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
 
   exchange_node_f (E,VN,lev);

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
