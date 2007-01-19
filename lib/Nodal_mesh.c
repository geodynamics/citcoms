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


void v_from_vector(E)
     struct All_variables *E;
{
    int i,eqn1,eqn2,eqn3,m,node;
    float sint,cost,sinf,cosf;

    const int nno = E->lmesh.nno;
    const int level=E->mesh.levmax;

    for (m=1;m<=E->sphere.caps_per_proc;m++)   {
        for(node=1;node<=nno;node++)     {
            E->sphere.cap[m].V[1][node] = E->U[m][E->id[m][node].doff[1]];
            E->sphere.cap[m].V[2][node] = E->U[m][E->id[m][node].doff[2]];
            E->sphere.cap[m].V[3][node] = E->U[m][E->id[m][node].doff[3]];
            if (E->node[m][node] & VBX)
                E->sphere.cap[m].V[1][node] = E->sphere.cap[m].VB[1][node];
            if (E->node[m][node] & VBY)
                E->sphere.cap[m].V[2][node] = E->sphere.cap[m].VB[2][node];
            if (E->node[m][node] & VBZ)
                E->sphere.cap[m].V[3][node] = E->sphere.cap[m].VB[3][node];
        }
        for (i=1;i<=E->lmesh.nno;i++)  {
            eqn1 = E->id[m][i].doff[1];
            eqn2 = E->id[m][i].doff[2];
            eqn3 = E->id[m][i].doff[3];
            sint = E->SinCos[level][m][0][i];
            sinf = E->SinCos[level][m][1][i];
            cost = E->SinCos[level][m][2][i];
            cosf = E->SinCos[level][m][3][i];
            E->temp[m][eqn1] = E->sphere.cap[m].V[1][i]*cost*cosf
                - E->sphere.cap[m].V[2][i]*sinf
                + E->sphere.cap[m].V[3][i]*sint*cosf;
            E->temp[m][eqn2] = E->sphere.cap[m].V[1][i]*cost*sinf
                + E->sphere.cap[m].V[2][i]*cosf
                + E->sphere.cap[m].V[3][i]*sint*sinf;
            E->temp[m][eqn3] = -E->sphere.cap[m].V[1][i]*sint
                + E->sphere.cap[m].V[3][i]*cost;

        }
    }

    return;
}

void v_from_vector_pseudo_surf(E)
     struct All_variables *E;
{
    int i,eqn1,eqn2,eqn3,m,node;
    float sint,cost,sinf,cosf;

    const int nno = E->lmesh.nno;
    const int level=E->mesh.levmax;
    double sum_V = 0.0, sum_dV = 0.0, rel_error = 0.0, global_max_error = 0.0;
    double tol_error = 1.0e-03;

    for (m=1;m<=E->sphere.caps_per_proc;m++)   {
        for(node=1;node<=nno;node++)     {
            E->sphere.cap[m].Vprev[1][node] = E->sphere.cap[m].V[1][node];
            E->sphere.cap[m].Vprev[2][node] = E->sphere.cap[m].V[2][node];
            E->sphere.cap[m].Vprev[3][node] = E->sphere.cap[m].V[3][node];

            E->sphere.cap[m].V[1][node] = E->U[m][E->id[m][node].doff[1]];
            E->sphere.cap[m].V[2][node] = E->U[m][E->id[m][node].doff[2]];
            E->sphere.cap[m].V[3][node] = E->U[m][E->id[m][node].doff[3]];
            if (E->node[m][node] & VBX)
                E->sphere.cap[m].V[1][node] = E->sphere.cap[m].VB[1][node];
            if (E->node[m][node] & VBY)
                E->sphere.cap[m].V[2][node] = E->sphere.cap[m].VB[2][node];
            if (E->node[m][node] & VBZ)
                E->sphere.cap[m].V[3][node] = E->sphere.cap[m].VB[3][node];

            sum_dV += (E->sphere.cap[m].V[1][node] - E->sphere.cap[m].Vprev[1][node])*(E->sphere.cap[m].V[1][node] - E->sphere.cap[m].Vprev[1][node])
                + (E->sphere.cap[m].V[2][node] - E->sphere.cap[m].Vprev[2][node])*(E->sphere.cap[m].V[2][node] - E->sphere.cap[m].Vprev[2][node])
                + (E->sphere.cap[m].V[3][node] - E->sphere.cap[m].Vprev[3][node])*(E->sphere.cap[m].V[3][node] - E->sphere.cap[m].Vprev[3][node]);
            sum_V += E->sphere.cap[m].V[1][node]*E->sphere.cap[m].V[1][node]
                + E->sphere.cap[m].V[2][node]*E->sphere.cap[m].V[2][node]
                + E->sphere.cap[m].V[3][node]*E->sphere.cap[m].V[3][node];
        }
        rel_error = sqrt(sum_dV)/sqrt(sum_V);
        MPI_Allreduce(&rel_error,&global_max_error,1,MPI_DOUBLE,MPI_MAX,E->parallel.world);
        if(global_max_error <= tol_error) E->monitor.stop_topo_loop = 1;
        if(E->parallel.me==0)
            fprintf(stderr,"global_max_error=%e stop_topo_loop=%d\n",global_max_error,E->monitor.stop_topo_loop);

        for (i=1;i<=E->lmesh.nno;i++)  {
            eqn1 = E->id[m][i].doff[1];
            eqn2 = E->id[m][i].doff[2];
            eqn3 = E->id[m][i].doff[3];
            sint = E->SinCos[level][m][0][i];
            sinf = E->SinCos[level][m][1][i];
            cost = E->SinCos[level][m][2][i];
            cosf = E->SinCos[level][m][3][i];
            E->temp[m][eqn1] = E->sphere.cap[m].V[1][i]*cost*cosf
                - E->sphere.cap[m].V[2][i]*sinf
                + E->sphere.cap[m].V[3][i]*sint*cosf;
            E->temp[m][eqn2] = E->sphere.cap[m].V[1][i]*cost*sinf
                + E->sphere.cap[m].V[2][i]*cosf
                + E->sphere.cap[m].V[3][i]*sint*sinf;
            E->temp[m][eqn3] = -E->sphere.cap[m].V[1][i]*sint
                + E->sphere.cap[m].V[3][i]*cost;

        }
    }

    return;
}


void velo_from_element(E,VV,m,el,sphere_key)
     struct All_variables *E;
     float VV[4][9];
     int m,el,sphere_key;
{

    int a, node;
    const int ends=enodes[E->mesh.nsd];

    if (sphere_key)
        for(a=1;a<=ends;a++)   {
            node = E->ien[m][el].node[a];
            VV[1][a] = E->sphere.cap[m].V[1][node];
            VV[2][a] = E->sphere.cap[m].V[2][node];
            VV[3][a] = E->sphere.cap[m].V[3][node];
        }
    else
        for(a=1;a<=ends;a++)   {
            node = E->ien[m][el].node[a];
            VV[1][a] = E->temp[m][E->id[m][node].doff[1]];
            VV[2][a] = E->temp[m][E->id[m][node].doff[2]];
            VV[3][a] = E->temp[m][E->id[m][node].doff[3]];
        }

    return;
}


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
