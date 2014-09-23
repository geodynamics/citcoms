/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
/* Functions relating to the building and use of mesh locations ... */


#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"



/* get nodal spherical velocities from the solution vector */
void v_from_vector(E)
     struct All_variables *E;
{
    int m,node;
    const int nno = E->lmesh.nno;

        for(node=1;node<=nno;node++)     {
            E->sphere.cap[CPPR].V[1][node] = E->U[E->id[node].doff[1]];
            E->sphere.cap[CPPR].V[2][node] = E->U[E->id[node].doff[2]];
            E->sphere.cap[CPPR].V[3][node] = E->U[E->id[node].doff[3]];
            if (E->node[node] & VBX)
                E->sphere.cap[CPPR].V[1][node] = E->sphere.cap[CPPR].VB[1][node];
            if (E->node[node] & VBY)
                E->sphere.cap[CPPR].V[2][node] = E->sphere.cap[CPPR].VB[2][node];
            if (E->node[node] & VBZ)
                E->sphere.cap[CPPR].V[3][node] = E->sphere.cap[CPPR].VB[3][node];
        }
}

void assign_v_to_vector(E)
     struct All_variables *E;
{
    int m,node;
    const int nno = E->lmesh.nno;

      for(node=1;node<=nno;node++)     {
	E->U[E->id[node].doff[1]] =  E->sphere.cap[CPPR].V[1][node];
	E->U[E->id[node].doff[2]] =  E->sphere.cap[CPPR].V[2][node];
	E->U[E->id[node].doff[3]] =  E->sphere.cap[CPPR].V[3][node];
      }
}

void v_from_vector_pseudo_surf(E)
     struct All_variables *E;
{
    int m,node;

    const int nno = E->lmesh.nno;
    double sum_V = 0.0, sum_dV = 0.0, rel_error = 0.0, global_max_error = 0.0;
    double tol_error = 1.0e-03;

        for(node=1;node<=nno;node++)     {
            E->sphere.cap[CPPR].Vprev[1][node] = E->sphere.cap[CPPR].V[1][node];
            E->sphere.cap[CPPR].Vprev[2][node] = E->sphere.cap[CPPR].V[2][node];
            E->sphere.cap[CPPR].Vprev[3][node] = E->sphere.cap[CPPR].V[3][node];

            E->sphere.cap[CPPR].V[1][node] = E->U[E->id[node].doff[1]];
            E->sphere.cap[CPPR].V[2][node] = E->U[E->id[node].doff[2]];
            E->sphere.cap[CPPR].V[3][node] = E->U[E->id[node].doff[3]];
            if (E->node[node] & VBX)
                E->sphere.cap[CPPR].V[1][node] = E->sphere.cap[CPPR].VB[1][node];
            if (E->node[node] & VBY)
                E->sphere.cap[CPPR].V[2][node] = E->sphere.cap[CPPR].VB[2][node];
            if (E->node[node] & VBZ)
                E->sphere.cap[CPPR].V[3][node] = E->sphere.cap[CPPR].VB[3][node];

            sum_dV += (E->sphere.cap[CPPR].V[1][node] - E->sphere.cap[CPPR].Vprev[1][node])*(E->sphere.cap[CPPR].V[1][node] - E->sphere.cap[CPPR].Vprev[1][node])
                + (E->sphere.cap[CPPR].V[2][node] - E->sphere.cap[CPPR].Vprev[2][node])*(E->sphere.cap[CPPR].V[2][node] - E->sphere.cap[CPPR].Vprev[2][node])
                + (E->sphere.cap[CPPR].V[3][node] - E->sphere.cap[CPPR].Vprev[3][node])*(E->sphere.cap[CPPR].V[3][node] - E->sphere.cap[CPPR].Vprev[3][node]);
            sum_V += E->sphere.cap[CPPR].V[1][node]*E->sphere.cap[CPPR].V[1][node]
                + E->sphere.cap[CPPR].V[2][node]*E->sphere.cap[CPPR].V[2][node]
                + E->sphere.cap[CPPR].V[3][node]*E->sphere.cap[CPPR].V[3][node];
        }
        rel_error = sqrt(sum_dV)/sqrt(sum_V);
        MPI_Allreduce(&rel_error,&global_max_error,1,MPI_DOUBLE,MPI_MAX,E->parallel.world);
        if(global_max_error <= tol_error) E->monitor.stop_topo_loop = 1;
        if(E->parallel.me==0)
            fprintf(stderr,"global_max_error=%e stop_topo_loop=%d\n",global_max_error,E->monitor.stop_topo_loop);

}

/* cartesian velocities within element, single prec version */
void velo_from_element(E,VV,el,sphere_key)
     struct All_variables *E;
     float VV[4][9];
     int el,sphere_key;
{

    int a, node;
    double sint, cost, sinf, cosf;
    const int ends=enodes[E->mesh.nsd];
    const int lev=E->mesh.levmax;

    if (sphere_key)
        for(a=1;a<=ends;a++)   {
            node = E->ien[el].node[a];
            VV[1][a] = E->sphere.cap[CPPR].V[1][node];
            VV[2][a] = E->sphere.cap[CPPR].V[2][node];
            VV[3][a] = E->sphere.cap[CPPR].V[3][node];
        }
    else {
        for(a=1;a<=ends;a++)   {
            node = E->ien[el].node[a];

            sint = E->SinCos[lev][CPPR][0][node]; 
            sinf = E->SinCos[lev][CPPR][1][node];
            cost = E->SinCos[lev][CPPR][2][node];
            cosf = E->SinCos[lev][CPPR][3][node];

            VV[1][a] = E->sphere.cap[CPPR].V[1][node]*cost*cosf
                - E->sphere.cap[CPPR].V[2][node]*sinf
                + E->sphere.cap[CPPR].V[3][node]*sint*cosf;
            VV[2][a] = E->sphere.cap[CPPR].V[1][node]*cost*sinf
                + E->sphere.cap[CPPR].V[2][node]*cosf
                + E->sphere.cap[CPPR].V[3][node]*sint*sinf;
            VV[3][a] = -E->sphere.cap[CPPR].V[1][node]*sint
                + E->sphere.cap[CPPR].V[3][node]*cost;
        }
    }
}

/* double prec version */
void velo_from_element_d(E,VV,el,sphere_key)
     struct All_variables *E;
     double VV[4][9];
     int el,sphere_key;
{

    int a, node;
    double sint, cost, sinf, cosf;
    const int dims=E->mesh.nsd;
    const int ends=enodes[E->mesh.nsd];
    const int nno=E->lmesh.nno;
    const int lev=E->mesh.levmax;

    if (sphere_key)
        for(a=1;a<=ends;a++)   {
            node = E->ien[el].node[a];
            VV[1][a] = E->sphere.cap[CPPR].V[1][node];
            VV[2][a] = E->sphere.cap[CPPR].V[2][node];
            VV[3][a] = E->sphere.cap[CPPR].V[3][node];
        }
    else {
        for(a=1;a<=ends;a++)   {
            node = E->ien[el].node[a];

            sint = E->SinCos[lev][CPPR][0][node];
            sinf = E->SinCos[lev][CPPR][1][node];
            cost = E->SinCos[lev][CPPR][2][node];
            cosf = E->SinCos[lev][CPPR][3][node];

            VV[1][a] = E->sphere.cap[CPPR].V[1][node]*cost*cosf
                - E->sphere.cap[CPPR].V[2][node]*sinf
                + E->sphere.cap[CPPR].V[3][node]*sint*cosf;
            VV[2][a] = E->sphere.cap[CPPR].V[1][node]*cost*sinf
                + E->sphere.cap[CPPR].V[2][node]*cosf
                + E->sphere.cap[CPPR].V[3][node]*sint*sinf;
            VV[3][a] = -E->sphere.cap[CPPR].V[1][node]*sint
                + E->sphere.cap[CPPR].V[3][node]*cost;
        }
    }
}

void p_to_nodes(E,P,PN,lev)
     struct All_variables *E;
     double *P;
     float *PN;
     int lev;

{ int e,element,node,j,m;

    for(node=1;node<=E->lmesh.NNO[lev];node++)
      PN[node] =  0.0;

    for(element=0;element<E->lmesh.NEL[lev];element++)
       for(j=1;j<=enodes[E->mesh.nsd];j++)  {
     	  node = E->IEN[lev][CPPR][element+1].node[j];
    	  PN[node] += P[element] * E->TWW[lev][CPPR][element+1].node[j] ;
    	  }

   (E->exchange_node_f)(E,PN,lev);

   for(node=1;node<=E->lmesh.NNO[lev];node++)
      PN[node] *= E->MASS[lev][node];
}


/* 

   interpolate the viscosity from element integration points to nodes

 */
void visc_from_gint_to_nodes(E,VE,VN,lev)
  struct All_variables *E;
  float *VE,*VN;
  int lev;
{
  int m,e,i,j,k,n,off,lim;
  const int nsd=E->mesh.nsd;
  const int vpts=vpoints[nsd];
  const int ends=enodes[nsd];
  double temp_visc;
  
    for(i=1;i<=E->lmesh.NNO[lev];i++)
      VN[i] = 0.0;
    for(e=1;e<=E->lmesh.NEL[lev];e++)   {
      temp_visc=0.0;
      for(i=1;i<=vpts;i++)
	temp_visc += VE[(e-1)*vpts + i];
      temp_visc = temp_visc/vpts;
      
      for(j=1;j<=ends;j++)                {
	n = E->IEN[lev][CPPR][e].node[j];
	VN[n] += E->TWW[lev][CPPR][e].node[j] * temp_visc;
      }
    }
  (E->exchange_node_f)(E,VN,lev);
    for(n=1;n<=E->lmesh.NNO[lev];n++)
      VN[n] *= E->MASS[lev][n];
}

/* 

interpolate viscosity from nodes to element integration points

 */
void visc_from_nodes_to_gint(E,VN,VE,lev)
  struct All_variables *E;
  float *VE,*VN;
  int lev;
{

  int m,e,i,j,k,n,off;
  const int nsd=E->mesh.nsd;
  const int vpts=vpoints[nsd];
  const int ends=enodes[nsd];
  double temp_visc;


    for(e=1;e<=E->lmesh.NEL[lev];e++)
      for(i=1;i<=vpts;i++)
	VE[(e-1)*vpts+i] = 0.0;
    for(e=1;e<=E->lmesh.NEL[lev];e++)
      for(i=1;i<=vpts;i++)      {
	temp_visc=0.0;
	for(j=1;j<=ends;j++)
	  temp_visc += E->N.vpt[GNVINDEX(j,i)]*VN[E->IEN[lev][CPPR][e].node[j]];
	VE[(e-1)*vpts+i] = temp_visc;
      }
}

/* called from MG as  (?)

   visc_from_gint_to_ele(E,E->EVI[lv],viscU,lv) 

*/
void visc_from_gint_to_ele(E,VE,VN,lev)
  struct All_variables *E;
  float *VE,*VN;
  int lev;
  {
    int m,e,i,j,k,n,off;
    const int nsd=E->mesh.nsd;
    const int vpts=vpoints[nsd];
    const int ends=enodes[nsd];
    double temp_visc;

      for(i=1;i<=E->lmesh.NEL[lev];i++)
	VN[i] = 0.0;
      for(e=1;e<=E->lmesh.NEL[lev];e++)   {
	temp_visc=0.0;
	for(i=1;i<=vpts;i++)
	  temp_visc += VE[(e-1)*vpts + i];
	temp_visc = temp_visc/vpts;
	VN[e] = temp_visc;
      }
}

/* called from MG as 

   visc_from_ele_to_gint(E,viscD,E->EVI[sl_minus],sl_minus); 

*/

void visc_from_ele_to_gint(E,VN,VE,lev)
  struct All_variables *E;
  float *VE,*VN;
  int lev;
{
  int m,e,i,j,k,n,off;
  const int nsd=E->mesh.nsd;
  const int vpts=vpoints[nsd];
  const int ends=enodes[nsd];
  double temp_visc;

    for(e=1;e<=E->lmesh.NEL[lev];e++)
      for(i=1;i<=vpts;i++)      {
	VE[(e-1)*vpts+i] = VN[e];
      }
}
