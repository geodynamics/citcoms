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
#include "element_definitions.h"
#include "global_defs.h"

static void geoid_from_buoyancy();
static void geoid_from_topography();


void get_STD_topo(E,tpg,tpgb,divg,vort,ii)
    struct All_variables *E;
    float **tpg,**tpgb;
    float **divg,**vort;
    int ii;
{
    void allocate_STD_mem();
    void compute_nodal_stress();
    void free_STD_mem();
    void get_surf_stress();

    int node,snode,m;
    float *SXX[NCS],*SYY[NCS],*SXY[NCS],*SXZ[NCS],*SZY[NCS],*SZZ[NCS];
    float *divv[NCS],*vorv[NCS];
    float topo_scaling1, topo_scaling2;

    allocate_STD_mem(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);
    compute_nodal_stress(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);

   if (E->parallel.me_loc[3]==E->parallel.nprocz-1)
     get_surf_stress(E,SXX,SYY,SZZ,SXY,SXZ,SZY);

/*    topo_scaling1=1.0/((E->data.density-E->data.density_above)*E->data.grav_acc); */
/*    topo_scaling2=1.0/((E->data.density_below-E->data.density)*E->data.grav_acc); */

   topo_scaling1 = topo_scaling2 = 1.0;

   for(m=1;m<=E->sphere.caps_per_proc;m++)
     for(snode=1;snode<=E->lmesh.nsf;snode++)   {
        node = E->surf_node[m][snode];
        tpg[m][snode]  = -2*SZZ[m][node] + SZZ[m][node-1];
        tpgb[m][snode] = 2*SZZ[m][node-E->lmesh.noz+1]-SZZ[m][node-E->lmesh.noz+2];
        tpg[m][snode]  = tpg[m][snode]*topo_scaling1;
        tpgb[m][snode]  = tpgb[m][snode]*topo_scaling2;

        divg[m][snode] = 2*divv[m][node]-divv[m][node-1];
        vort[m][snode] = 2*vorv[m][node]-vorv[m][node-1];
     }

   free_STD_mem(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);

   return;
}

void get_STD_freesurf(struct All_variables *E,float **freesurf)
{
        int node,snode,m;

        if (E->parallel.me_loc[3]==E->parallel.nprocz-1)
                for(m=1;m<=E->sphere.caps_per_proc;m++)
                        for(snode=1;snode<=E->lmesh.nsf;snode++) {
                                node = E->surf_node[m][snode];
                                /*freesurf[m][snode] += 0.5*(E->sphere.cap[m].V[3][node]+E->sphere.cap[m].Vprev[3][node])*E->advection.timestep;*/
                                freesurf[m][snode] += E->sphere.cap[m].V[3][node]*E->advection.timestep;
                        }
        return;
}


void allocate_STD_mem(struct All_variables *E,
                      float** SXX, float** SYY, float** SZZ,
                      float** SXY, float** SXZ, float** SZY,
                      float** divv, float** vorv)
{
  int m, i;

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    SXX[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
    SYY[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
    SXY[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
    SXZ[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
    SZY[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
    SZZ[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
    divv[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
    vorv[m] = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
  }

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    for(i=1;i<=E->lmesh.nno;i++) {
      SZZ[m][i] = 0.0;
      SXX[m][i] = 0.0;
      SYY[m][i] = 0.0;
      SXY[m][i] = 0.0;
      SXZ[m][i] = 0.0;
      SZY[m][i] = 0.0;
      divv[m][i] = 0.0;
      vorv[m][i] = 0.0;
    }
  }
  return;
}


void free_STD_mem(struct All_variables *E,
                  float** SXX, float** SYY, float** SZZ,
                  float** SXY, float** SXZ, float** SZY,
                  float** divv, float** vorv)
{
  int m;
  for(m=1;m<=E->sphere.caps_per_proc;m++)        {
    free((void *)SXX[m]);
    free((void *)SYY[m]);
    free((void *)SXY[m]);
    free((void *)SXZ[m]);
    free((void *)SZY[m]);
    free((void *)SZZ[m]);
    free((void *)divv[m]);
    free((void *)vorv[m]);
    }
}


void get_surf_stress(E,SXX,SYY,SZZ,SXY,SXZ,SZY)
  struct All_variables *E;
  float **SXX,**SYY,**SZZ,**SXY,**SXZ,**SZY;
{
  int m,i,node,stride;

  stride = E->lmesh.nsf*6;

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for (node=1;node<=E->lmesh.nno;node++)
      if ( (node%E->lmesh.noz)==0 )  {
        i = node/E->lmesh.noz;
        E->stress[m][(i-1)*6+1] = SXX[m][node];
        E->stress[m][(i-1)*6+2] = SYY[m][node];
        E->stress[m][(i-1)*6+3] = SZZ[m][node];
        E->stress[m][(i-1)*6+4] = SXY[m][node];
        E->stress[m][(i-1)*6+5] = SXZ[m][node];
        E->stress[m][(i-1)*6+6] = SZY[m][node];
        }
     else if ( ((node+1)%E->lmesh.noz)==0 )  {
        i = (node+1)/E->lmesh.noz;
        E->stress[m][stride+(i-1)*6+1] = SXX[m][node];
        E->stress[m][stride+(i-1)*6+2] = SYY[m][node];
        E->stress[m][stride+(i-1)*6+3] = SZZ[m][node];
        E->stress[m][stride+(i-1)*6+4] = SXY[m][node];
        E->stress[m][stride+(i-1)*6+5] = SXZ[m][node];
        E->stress[m][stride+(i-1)*6+6] = SZY[m][node];
        }

  return;
}


void compute_nodal_stress(struct All_variables *E,
                          float** SXX, float** SYY, float** SZZ,
                          float** SXY, float** SXZ, float** SZY,
                          float** divv, float** vorv)
{
  void get_global_shape_fn();
  void velo_from_element();
  void stress_conform_bcs();

  int i,j,e,node,m;

  float VV[4][9],Vxyz[9][9],Szz,Sxx,Syy,Sxy,Sxz,Szy,div,vor;
  double pre[9],tww[9],rtf[4][9];
  double velo_scaling, stress_scaling;

  struct Shape_function GN;
  struct Shape_function_dA dOmega;
  struct Shape_function_dx GNx;

  const int dims=E->mesh.nsd;
  const int vpts=vpoints[dims];
  const int ends=enodes[dims];
  const int lev=E->mesh.levmax;
  const int sphere_key=1;

  for(m=1;m<=E->sphere.caps_per_proc;m++) {
    for(e=1;e<=E->lmesh.nel;e++)  {
      Szz = 0.0;
      Sxx = 0.0;
      Syy = 0.0;
      Sxy = 0.0;
      Sxz = 0.0;
      Szy = 0.0;
      div = 0.0;
      vor = 0.0;
      get_global_shape_fn(E,e,&GN,&GNx,&dOmega,0,sphere_key,rtf,E->mesh.levmax,m);

      velo_from_element(E,VV,m,e,sphere_key);

      for(j=1;j<=vpts;j++)  {
        pre[j] =  E->EVi[m][(e-1)*vpts+j]*dOmega.vpt[j];
        Vxyz[1][j] = 0.0;
        Vxyz[2][j] = 0.0;
        Vxyz[3][j] = 0.0;
        Vxyz[4][j] = 0.0;
        Vxyz[5][j] = 0.0;
        Vxyz[6][j] = 0.0;
        Vxyz[7][j] = 0.0;
        Vxyz[8][j] = 0.0;
      }

      for(i=1;i<=ends;i++) {
        tww[i] = 0.0;
        for(j=1;j<=vpts;j++)
          tww[i] += dOmega.vpt[j] * g_point[j].weight[E->mesh.nsd-1]
            * E->N.vpt[GNVINDEX(i,j)];
      }

      for(j=1;j<=vpts;j++)   {
        for(i=1;i<=ends;i++)   {
          Vxyz[1][j]+=( VV[1][i]*GNx.vpt[GNVXINDEX(0,i,j)]
                        + VV[3][i]*E->N.vpt[GNVINDEX(i,j)] )*rtf[3][j];
          Vxyz[2][j]+=( (VV[2][i]*GNx.vpt[GNVXINDEX(1,i,j)]
                         + VV[1][i]*E->N.vpt[GNVINDEX(i,j)]*cos(rtf[1][j]))/sin(rtf[1][j])
                        + VV[3][i]*E->N.vpt[GNVINDEX(i,j)] )*rtf[3][j];
          Vxyz[3][j]+= VV[3][i]*GNx.vpt[GNVXINDEX(2,i,j)];

          Vxyz[4][j]+=( (VV[1][i]*GNx.vpt[GNVXINDEX(1,i,j)]
                         - VV[2][i]*E->N.vpt[GNVINDEX(i,j)]*cos(rtf[1][j]))/sin(rtf[1][j])
                        + VV[2][i]*GNx.vpt[GNVXINDEX(0,i,j)])*rtf[3][j];
          Vxyz[5][j]+=VV[1][i]*GNx.vpt[GNVXINDEX(2,i,j)] + rtf[3][j]*(VV[3][i]
                                                                      *GNx.vpt[GNVXINDEX(0,i,j)]-VV[1][i]*E->N.vpt[GNVINDEX(i,j)]);
          Vxyz[6][j]+=VV[2][i]*GNx.vpt[GNVXINDEX(2,i,j)] + rtf[3][j]*(VV[3][i]
                                                                      *GNx.vpt[GNVXINDEX(1,i,j)]/sin(rtf[1][j])-VV[2][i]*E->N.vpt[GNVINDEX(i,j)]);
          Vxyz[7][j]+=rtf[3][j] * (
                                   VV[1][i]*GNx.vpt[GNVXINDEX(0,i,j)]
                                   + VV[1][i]*E->N.vpt[GNVINDEX(i,j)]*cos(rtf[1][j])/sin(rtf[1][j])
                                   + VV[2][i]*GNx.vpt[GNVXINDEX(1,i,j)]/sin(rtf[1][j])  );
          Vxyz[8][j]+=rtf[3][j]/sin(rtf[1][j])*
            ( VV[2][i]*GNx.vpt[GNVXINDEX(0,i,j)]*sin(rtf[1][j])
              + VV[2][i]*E->N.vpt[GNVINDEX(i,j)]*cos(rtf[1][j])
              - VV[1][i]*GNx.vpt[GNVXINDEX(1,i,j)] );
        }
        Sxx += 2.0 * pre[j] * Vxyz[1][j];
        Syy += 2.0 * pre[j] * Vxyz[2][j];
        Szz += 2.0 * pre[j] * Vxyz[3][j];
        Sxy += pre[j] * Vxyz[4][j];
        Sxz += pre[j] * Vxyz[5][j];
        Szy += pre[j] * Vxyz[6][j];
        div += Vxyz[7][j]*dOmega.vpt[j];
        vor += Vxyz[8][j]*dOmega.vpt[j];
      }

      Sxx /= E->eco[m][e].area;
      Syy /= E->eco[m][e].area;
      Szz /= E->eco[m][e].area;
      Sxy /= E->eco[m][e].area;
      Sxz /= E->eco[m][e].area;
      Szy /= E->eco[m][e].area;
      div /= E->eco[m][e].area;
      vor /= E->eco[m][e].area;

      Szz -= E->P[m][e];  /* add the pressure term */
      Sxx -= E->P[m][e];  /* add the pressure term */
      Syy -= E->P[m][e];  /* add the pressure term */

      for(i=1;i<=ends;i++) {
        node = E->ien[m][e].node[i];
        SZZ[m][node] += tww[i] * Szz;
        SXX[m][node] += tww[i] * Sxx;
        SYY[m][node] += tww[i] * Syy;
        SXY[m][node] += tww[i] * Sxy;
        SXZ[m][node] += tww[i] * Sxz;
        SZY[m][node] += tww[i] * Szy;
        divv[m][node]+= tww[i] * div;
        vorv[m][node]+= tww[i] * vor;
      }

    }    /* end for el */
  }     /* end for m */

  (E->exchange_node_f)(E,SXX,lev);
  (E->exchange_node_f)(E,SYY,lev);
  (E->exchange_node_f)(E,SZZ,lev);
  (E->exchange_node_f)(E,SXY,lev);
  (E->exchange_node_f)(E,SXZ,lev);
  (E->exchange_node_f)(E,SZY,lev);
  (E->exchange_node_f)(E,divv,lev);
  (E->exchange_node_f)(E,vorv,lev);

  /*    stress_scaling = 1.0e-6*E->data.ref_viscosity*E->data.therm_diff/ */
  /*                       (E->data.radius_km*E->data.radius_km); */

  /*    velo_scaling = 100.*365.*24.*3600.*1.0e-3*E->data.therm_diff/E->data.radius_km; */
  /* cm/yr */

  stress_scaling = velo_scaling = 1.0;

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(node=1;node<=E->lmesh.nno;node++)   {
      SZZ[m][node] = SZZ[m][node]*E->Mass[m][node]*stress_scaling;
      SXX[m][node] = SXX[m][node]*E->Mass[m][node]*stress_scaling;
      SYY[m][node] = SYY[m][node]*E->Mass[m][node]*stress_scaling;
      SXY[m][node] = SXY[m][node]*E->Mass[m][node]*stress_scaling;
      SXZ[m][node] = SXZ[m][node]*E->Mass[m][node]*stress_scaling;
      SZY[m][node] = SZY[m][node]*E->Mass[m][node]*stress_scaling;
      vorv[m][node] = vorv[m][node]*E->Mass[m][node]*velo_scaling;
      divv[m][node] = divv[m][node]*E->Mass[m][node]*velo_scaling;
    }

  /* assign stress to all the nodes */
  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for (node=1;node<=E->lmesh.nno;node++) {
      E->gstress[m][(node-1)*6+1] = SXX[m][node];
      E->gstress[m][(node-1)*6+2] = SYY[m][node];
      E->gstress[m][(node-1)*6+3] = SZZ[m][node];
      E->gstress[m][(node-1)*6+4] = SXY[m][node];
      E->gstress[m][(node-1)*6+5] = SXZ[m][node];
      E->gstress[m][(node-1)*6+6] = SZY[m][node];
    }

  /* replace boundary stresses with boundary conditions (if specified) */
  stress_conform_bcs(E);

}




void stress_conform_bcs(struct All_variables *E)
{
  int m, i, j, k, n, d;
  const unsigned sbc_flag[4] = {0, SBX, SBY, SBZ};
  const int stress_index[4][4] = { {0, 0, 0, 0},
                                   {0, 1, 4, 5},
                                   {0, 4, 2, 6},
                                   {0, 5, 6, 3} };

  if(E->control.side_sbcs) {

    for(m=1; m<=E->sphere.caps_per_proc; m++)
      for(i=1; i<=E->lmesh.noy; i++)
        for(j=1; j<=E->lmesh.nox; j++)
          for(k=1; k<=E->lmesh.noz; k++) {
            n = k+(j-1)*E->lmesh.noz+(i-1)*E->lmesh.nox*E->lmesh.noz;
            for(d=1; d<=E->mesh.nsd; d++)
              if(E->node[m][n] & sbc_flag[d]) {
                if(i==1)
                  E->gstress[m][(n-1)*6+stress_index[d][2]] = E->sbc.SB[m][SIDE_WEST][d][ E->sbc.node[m][n] ];
                if(i==E->lmesh.noy)
                  E->gstress[m][(n-1)*6+stress_index[d][2]] = E->sbc.SB[m][SIDE_EAST][d][ E->sbc.node[m][n] ];
                if(j==1)
                  E->gstress[m][(n-1)*6+stress_index[d][1]] = E->sbc.SB[m][SIDE_NORTH][d][ E->sbc.node[m][n] ];
                if(j==E->lmesh.nox)
                  E->gstress[m][(n-1)*6+stress_index[d][1]] = E->sbc.SB[m][SIDE_SOUTH][d][ E->sbc.node[m][n] ];
                if(k==1)
                  E->gstress[m][(n-1)*6+stress_index[d][3]] = E->sbc.SB[m][SIDE_BOTTOM][d][ E->sbc.node[m][n] ];
                if(k==E->lmesh.noz)
                  E->gstress[m][(n-1)*6+stress_index[d][3]] = E->sbc.SB[m][SIDE_TOP][d][ E->sbc.node[m][n] ];
              }
          }

  } else {

    for(m=1; m<=E->sphere.caps_per_proc; m++)
      for(i=1; i<=E->lmesh.noy; i++)
        for(j=1; j<=E->lmesh.nox; j++)
          for(k=1; k<=E->lmesh.noz; k++) {
            n = k+(j-1)*E->lmesh.noz+(i-1)*E->lmesh.nox*E->lmesh.noz;
            for(d=1; d<=E->mesh.nsd; d++)
              if(E->node[m][n] & sbc_flag[d]) {
                if(i==1 || i==E->lmesh.noy)
                  E->gstress[m][(n-1)*6+stress_index[d][2]] = E->sphere.cap[m].VB[d][n];
                if(j==1 || j==E->lmesh.nox)
                  E->gstress[m][(n-1)*6+stress_index[d][1]] = E->sphere.cap[m].VB[d][n];
                if(k==1 || k==E->lmesh.noz)
                  E->gstress[m][(n-1)*6+stress_index[d][3]] = E->sphere.cap[m].VB[d][n];
              }
          }
  }
}


/* ===================================================================
   ===================================================================  */
void compute_geoid(E, harm_geoid, harm_tpgt, harm_tpgb)
     struct All_variables *E;
     float *harm_geoid[2], *harm_tpgt[2], *harm_tpgb[2];
{
    int ll, mm, p;

    for (ll=1;ll<=E->output.llmax;ll++)
        for (mm=0;mm<=ll;mm++)   {
            p = E->sphere.hindex[ll][mm];
            harm_geoid[0][p] = 0.0;
            harm_geoid[1][p] = 0.0;
        }

    geoid_from_buoyancy(E, harm_geoid);
    geoid_from_topography(E, harm_geoid, harm_tpgt, harm_tpgb);
}


static void geoid_from_buoyancy(E, harm_geoid)
     struct All_variables *E;
     float *harm_geoid[2];
{
    int m,k,ll,mm,node,i,j,p,noz,snode;
    float *TT[NCS],radius,*geoid[2],dlayer,con1,con2,scaling2,scaling;
    void sphere_expansion();
    void sum_across_depth_sph1();

    /* scale for buoyancy */
    scaling2 = -E->data.therm_exp*E->data.ref_temperature*E->data.density/E->control.Atemp;
    /* scale for geoid */
    scaling = 1.0e6*4.0*M_PI*E->data.radius_km*E->data.radius_km*
        E->data.grav_const/E->data.grav_acc;

    /* buoyancy of one layer */
    for(m=1;m<=E->sphere.caps_per_proc;m++)
        TT[m] = (float *) malloc ((E->lmesh.nsf+1)*sizeof(float));

    /* sin coeff */
    geoid[0] = (float*)malloc((E->sphere.hindice+1)*sizeof(float));
    /* cos coeff */
    geoid[1] = (float*)malloc((E->sphere.hindice+1)*sizeof(float));

    /* loop over each layer */
    for(k=1;k<E->lmesh.noz;k++)  {

        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=E->lmesh.noy;i++)
                for(j=1;j<=E->lmesh.nox;j++)  {
                    node=k+(j-1)*E->lmesh.noz+(i-1)*E->lmesh.nox*E->lmesh.noz;
                    p = j + (i-1)*E->lmesh.nox;
                    TT[m][p] = (E->buoyancy[m][node]+E->buoyancy[m][node+1])*0.5*scaling2;
                }

        /* expand TT into spherical harmonics */
        sphere_expansion(E,TT,geoid[0],geoid[1]);

        /* thickness of the layer */
        dlayer = (E->sx[1][3][k+1]-E->sx[1][3][k]);

        /* radius of the layer */
        radius = (E->sx[1][3][k+1]+E->sx[1][3][k])*0.5;

        /* contribution of buoyancy at this layer */
        for (ll=1;ll<=E->output.llmax;ll++)
            for (mm=0;mm<=ll;mm++)   {
                con1 = scaling/(2.0*ll+1.0)*dlayer;
                con2 = pow(radius,((double)(ll+2)));

                p = E->sphere.hindex[ll][mm];
                harm_geoid[0][p] += con1*con2*geoid[0][p];
                harm_geoid[1][p] += con1*con2*geoid[1][p];
            }

        //if(E->parallel.me==0)  fprintf(stderr,"layer %d %.5e %g %g %g\n",k,radius,dlayer,con1,con2);
    }

    /* accumulate geoid from all layers to the surface (top processors) */
    sum_across_depth_sph1(E, harm_geoid[0], harm_geoid[1]);

    for(m=1;m<=E->sphere.caps_per_proc;m++)
        free ((void *)TT[m]);

    free ((void *)geoid[0]);
    free ((void *)geoid[1]);
    return;
}



static void geoid_from_topography(E, harm_geoid, harm_tpgt, harm_tpgb)
     struct All_variables *E;
     float *harm_geoid[2], *harm_tpgt[2], *harm_tpgb[2];
{

    float *geoid[2],con1,con2,scaling,den_contrast1,den_contrast2,stress_scaling,topo_scaling1,topo_scaling2;
    int i,j,k,ll,mm,s;
    void sum_across_depth_sph1();
    void sphere_expansion();

    geoid[0] = (float *)malloc((E->sphere.hindice+2)*sizeof(float));
    geoid[1] = (float *)malloc((E->sphere.hindice+2)*sizeof(float));

    for (i=0;i<E->sphere.hindice;i++)   {
        geoid[0][i] = 0.0;
        geoid[1][i] = 0.0;
    }

    stress_scaling = E->data.ref_viscosity*E->data.therm_diff/
        (E->data.radius_km*E->data.radius_km*1e6);

    /* density across surface */
    den_contrast1 = (E->data.density-E->data.density_above);
    /* density across CMB */
    den_contrast2 = (E->data.density_below-E->data.density);

    /* scale for surface and CMB topo */
    topo_scaling1 = stress_scaling / (den_contrast1*E->data.grav_acc);
    topo_scaling2 = stress_scaling / (den_contrast2*E->data.grav_acc);

    /* scale for geoid */
    scaling = 1.0e3*4.0*M_PI*E->data.radius_km*
        E->data.grav_const/E->data.grav_acc;


    if (E->parallel.me_loc[3]==E->parallel.nprocz-1) {
        /* dimensionalize surface topography and
           expand into spherical harmonics */
        for(j=1;j<=E->sphere.caps_per_proc;j++)
            for(i=1;i<=E->lmesh.nsf;i++)   {
                E->slice.tpg[j][i] = E->slice.tpg[j][i]*topo_scaling1;
            }
        sphere_expansion(E,E->slice.tpg,E->sphere.harm_tpgt[0],
                         E->sphere.harm_tpgt[1]);
    }
    //print_field_spectral_regular(E,TL,E->sphere.harm_tpgt[0],E->sphere.harm_tpgt[1],E->parallel.nprocz-1,0,"tpgt");




    if (E->parallel.me_loc[3]==0) {
        /* dimensionalize CMB topography and
           expand into spherical harmonics */
        for(j=1;j<=E->sphere.caps_per_proc;j++)
            for(i=1;i<=E->lmesh.nsf;i++)   {
                E->slice.tpgb[j][i] = E->slice.tpgb[j][i]*topo_scaling2;
            }
        sphere_expansion(E,E->slice.tpgb,E->sphere.harm_tpgb[0],
                         E->sphere.harm_tpgb[1]);
    }
    //print_field_spectral_regular(E,TL,E->sphere.harm_tpgb[0],E->sphere.harm_tpgb[1],0,0,"tpgb");





    if (E->parallel.me_loc[3] == E->parallel.nprocz-1)
        for (ll=2;ll<=E->output.llmax;ll++)   {
            con1 = scaling/(2.0*ll+1.0);
            for (mm=0;mm<=ll;mm++)   {
                i = E->sphere.hindex[ll][mm];
                geoid[0][i]+= E->sphere.harm_tpgt[0][i]*con1*den_contrast1;
                geoid[1][i]+= E->sphere.harm_tpgt[1][i]*con1*den_contrast1;
            }
        }




    if (E->parallel.me_loc[3] == 0)
        for (ll=2;ll<=E->output.llmax;ll++)   {
            con1 = scaling/(2.0*ll+1.0);
            con2 = pow(E->sphere.ri,((double)(ll+2)));
            for (mm=0;mm<=ll;mm++)   {
                i = E->sphere.hindex[ll][mm];
                geoid[0][i]+= E->sphere.harm_tpgb[0][i]*con1*con2*den_contrast2;
                geoid[1][i]+= E->sphere.harm_tpgb[1][i]*con1*con2*den_contrast2;
            }
        }

    /* accumulate geoid to the surface (top processors) */
    sum_across_depth_sph1(E,geoid[0],geoid[1]);

    // add to the geoid coeff
    for (ll=2;ll<=E->output.llmax;ll++)
        for (mm=0;mm<=ll;mm++)   {
            i = E->sphere.hindex[ll][mm];
            harm_geoid[0][i]+= geoid[0][i];
            harm_geoid[1][i]+= geoid[1][i];
        }



#if 0
    E->sphere.harm_geoid[0][0]=0.0;
    E->sphere.harm_geoid[1][0]=0.0;
    E->sphere.harm_geoid[0][1]=0.0;
    E->sphere.harm_geoid[1][1]=0.0;
    E->sphere.harm_geoid[0][2]=0.0;
    E->sphere.harm_geoid[1][2]=0.0;  /* zero the 00 10 11 */

    inv_sphere_harmonics(E,E->sphere.harm_geoid[0],E->sphere.harm_geoid[1],TL,E->parallel.nprocz-1);

    print_field_spectral_regular(E,TL,E->sphere.harm_geoid[0],E->sphere.harm_geoid[1],E->parallel.nprocz-1,1,"geoid");
#endif



    free ((void *)geoid[0]);
    free ((void *)geoid[1]);

    return;
}

/* ===================================================================
   Consistent boundary flux method for stress ... Zhong,Gurnis,Hulbert

   Solve for the stress as the code defined it internally, rather than
   what was intended to be solved. This is more appropriate.

   Note also that the routine is dependent on the method
   used to solve for the velocity in the first place.
   ===================================================================  */

void get_CBF_topo(E,H,HB)       /* call this only for top and bottom processors*/
    struct All_variables *E;
    float **H,**HB;

{
/*     void get_elt_k(); */
/*     void get_elt_g(); */
/*     void get_elt_f(); */
/*     void get_global_1d_shape_fn(); */
/*     void exchange_snode_f(); */
/*     void velo_from_element(); */

/*     int a,address,el,elb,els,node,nodeb,nodes,i,j,k,l,m,n,count; */
/*     int nodel,nodem,nodesl,nodesm,nnsf,nel2; */

/*     struct Shape_function1 GM,GMb; */
/*     struct Shape_function1_dA dGammax,dGammabx; */

/*     float *eltTU,*eltTL,*SU[NCS],*SL[NCS],*RU[NCS],*RL[NCS]; */
/*     float VV[4][9]; */

/*     double eltk[24*24],eltf[24]; */
/*     double eltkb[24*24],eltfb[24]; */
/*     double res[24],resb[24],eu[24],eub[24]; */
/*     higher_precision eltg[24][1],eltgb[24][1]; */

/*     const int dims=E->mesh.nsd; */
/*     const int Tsize=5;   */ /* maximum values, applicable to 3d, harmless for 2d */
/*     const int Ssize=4; */
/*     const int ends=enodes[dims]; */
/*     const int noz=E->lmesh.noz; */
/*     const int noy=E->lmesh.noy; */
/*     const int nno=E->lmesh.nno; */
/*     const int onedv=onedvpoints[dims]; */
/*     const int snode1=1,snode2=4,snode3=5,snode4=8; */
/*     const int elz = E->lmesh.elz; */
/*     const int ely = E->lmesh.ely; */
/*     const int lev=E->mesh.levmax; */
/*     const int sphere_key=1; */

/*     const int lnsf=E->lmesh.nsf; */

/*     eltTU = (float *)malloc((1+Tsize)*sizeof(float));  */
/*     eltTL = (float *)malloc((1+Tsize)*sizeof(float)); */

/*   for(j=1;j<=E->sphere.caps_per_proc;j++)          { */
/*     SU[j] = (float *)malloc((1+lnsf)*sizeof(float)); */
/*     SL[j] = (float *)malloc((1+lnsf)*sizeof(float)); */
/*     RU[j] = (float *)malloc((1+lnsf)*sizeof(float)); */
/*     RL[j] = (float *)malloc((1+lnsf)*sizeof(float)); */
/*     } */

/*   for(j=1;j<=E->sphere.caps_per_proc;j++)          { */

/*     for(i=0;i<=lnsf;i++) */
/*       RU[j][i] = RL[j][i] = SU[j][i] = SL[j][i] = 0.0; */

    /* calculate the element residuals */

/*     for(els=1;els<=E->lmesh.snel;els++) { */
/*       el = E->surf_element[j][els]; */
/*       elb = el + elz-1; */

/*       for(m=0;m<ends;m++) { */  /* for bottom elements no faults */
/*           nodeb= E->ien[j][elb].node[m+1]; */
/*           eub[m*dims  ] = E->sphere.cap[j].V[1][nodeb]; */
/*           eub[m*dims+1] = E->sphere.cap[j].V[2][nodeb]; */
/*           eub[m*dims+2] = E->sphere.cap[j].V[3][nodeb];  */
/*           } */

/*       velo_from_element(E,VV,j,el,sphere_key); */

/*       for(m=0;m<ends;m++) {   */
/*          eu [m*dims  ] = VV[1][m+1]; */
/*          eu [m*dims+1] = VV[2][m+1]; */
/*          eu [m*dims+2] = VV[3][m+1]; */
/*          } */

/*       get_elt_k(E,el,eltk,lev,j); */
/*       get_elt_k(E,elb,eltkb,lev,j); */
/*       get_elt_f(E,el,eltf,0,j); */
/*       get_elt_f(E,elb,eltfb,0,j); */
/*       get_elt_g(E,el,eltg,lev,j); */
/*       get_elt_g(E,elb,eltgb,lev,j); */

/*       for(m=0;m<dims*ends;m++) { */
/*             res[m]  = eltf[m]  - E->elt_del[lev][j][el].g[m][0]  * E->P[j][el]; */
/*             resb[m] = eltfb[m] - E->elt_del[lev][j][elb].g[m][0]* E->P[j][elb]; */
/*             } */

/*       for(m=0;m<dims*ends;m++) */
/*          for(l=0;l<dims*ends;l++) { */
/*               res[m]  -= eltk[ends*dims*m+l]  * eu[l]; */
/*               resb[m] -= eltkb[ends*dims*m+l] * eub[l]; */
/*               } */

            /* Put relevant (vertical & surface) parts of element residual into surface residual */

/*       for(m=1;m<=ends;m++) {    */  /* for bottom elements */
/*          switch (m) { */
/*              case 2: */
/*                  RL[j][E->sien[j][els].node[1]] += resb[(m-1)*dims+1];   */
/*                  break; */
/*              case 3: */
/*                  RL[j][E->sien[j][els].node[2]] += resb[(m-1)*dims+1];   */
/*                  break; */
/*              case 7: */
/*                  RL[j][E->sien[j][els].node[3]] += resb[(m-1)*dims+1];   */
/*                  break; */
/*              case 6: */
/*                  RL[j][E->sien[j][els].node[4]] += resb[(m-1)*dims+1];   */
/*                  break; */
/*                  } */
/*              } */


/*       for(m=1;m<=ends;m++) { */
/*          switch (m) { */
/*              case 1: */
/*                 nodes = E->sien[j][els].node[1]; */
/*                 break; */
/*              case 4: */
/*                 nodes = E->sien[j][els].node[2]; */
/*                 break; */
/*              case 8: */
/*                 nodes = E->sien[j][els].node[3]; */
/*                 break; */
/*              case 5: */
/*                 nodes = E->sien[j][els].node[4]; */
/*                 break; */
/*              } */

/*              RU[j][nodes] += res[(m-1)*dims+1];   */
/*          }  */     /* end for m */
/*       } */

    /* calculate the LHS */

/*     for(els=1;els<=E->lmesh.snel;els++) { */
/*        el = E->surf_element[j][els]; */
/*        elb = el + elz-1; */

/*        get_global_1d_shape_fn(E,el,&GM,&dGammax,0,j); */
/*        get_global_1d_shape_fn(E,elb,&GMb,&dGammabx,0,j); */

/*        for(m=1;m<=onedv;m++)        { */
/*           eltTU[m-1] = 0.0; */
/*           eltTL[m-1] = 0.0;  */
/*           for(n=1;n<=onedv;n++)          { */
/*              eltTU[m-1] +=  */
/*                 dGammax.vpt[GMVGAMMA(1,n)] * l_1d[n].weight[dims-1] */
/*                 * E->L.vpt[GMVINDEX(m,n)] * E->L.vpt[GMVINDEX(m,n)]; */
/*              eltTL[m-1] +=  */
/*                      dGammabx.vpt[GMVGAMMA(1+dims,n)]*l_1d[n].weight[dims-1] */
/*                 * E->L.vpt[GMVINDEX(m,n)] * E->L.vpt[GMVINDEX(m,n)]; */
/*              } */
/*           } */

/*         for (m=1;m<=onedv;m++)  */    /* for bottom */
/*             SL[m][E->sien[m][els].node[m]] += eltTL[m-1]; */

/*         for (m=1;m<=onedv;m++)  { */
/*             if (m==1)  */
/*                 a = 1; */
/*             else if (m==2) */
/*                 a = 4; */
/*             else if (m==3) */
/*                 a = 8; */
/*             else if (m==4) */
/*                 a = 5; */

/*             nodes = E->sien[m][els].node[m]; */
/*             SU[m][E->sien[m][els].node[m]] += eltTU[m-1]; */
/*             } */
/*         } */

/*       }  */     /* end for j */


/*     if (E->parallel.me_loc[3]==0)  {      */    /* for top topography */
/*       for(i=1;i<=E->lmesh.nsf;i++)   */

/*       exchange_snode_f(E,RU,SU,E->mesh.levmax); */

/*       for (j=1;j<=E->sphere.caps_per_proc;j++) */
/*         for(i=1;i<=E->lmesh.nsf;i++) */
/*           H[j][i] = -RU[j][i]/SU[j][i]; */
/*       } */

/*     if (E->parallel.me_loc[3]==E->parallel.nprocz-1)   {  */   /* for bottom topo */
/*       exchange_snode_f(E,RL,SL,E->mesh.levmax); */
/*       for (j=1;j<=E->sphere.caps_per_proc;j++) */
/*         for(i=1;i<=E->lmesh.nsf;i++) */
/*           HB[j][i] = -RL[j][i]/SL[j][i]; */
/*       } */

/*     free((void *)eltTU); */
/*     free((void *)eltTL); */
/*     for (j=1;j<=E->sphere.caps_per_proc;j++)   { */
/*       free((void *)SU[j]); */
/*       free((void *)SL[j]); */
/*       free((void *)RU[j]); */
/*       free((void *)RL[j]); */
/*       } */
    return;
 }


/* version */
/* $Id$ */

/* End of file  */
