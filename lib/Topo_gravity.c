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

void myerror(struct All_variables *, char *);
void sphere_expansion(struct All_variables *, float *, float *, float *);
void sum_across_depth_sph1(struct All_variables *, float *, float *);
void broadcast_vertical(struct All_variables *, float *, float *, int);
long double lg_pow(long double, int);
void allocate_STD_mem(struct All_variables *E,
                      float* , float* , float* ,
                      float* , float* , float* ,
                      float* , float* );
void free_STD_mem(struct All_variables *E,
                  float* , float* , float* ,
                  float* , float* , float* ,
                  float* , float* );
void compute_nodal_stress(struct All_variables *,
                          float* , float* , float* ,
                          float* , float* , float* ,
                          float* , float* );
void stress_conform_bcs(struct All_variables *);
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
#include "anisotropic_viscosity.h"
#endif


/* 

compute the full stress tensor and the dynamic topo

here, we only need szz, but leave in for potential stress output if
removed, make sure to recompute in output routines


 */

void get_STD_topo(E,tpg,tpgb,divg,vort,ii)
    struct All_variables *E;
    float *tpg,*tpgb;
    float *divg,*vort;
    int ii;
{
    void allocate_STD_mem();
    void compute_nodal_stress();
    void free_STD_mem();
    //void get_surf_stress();

    int node,snode,m;
    float *SXX,*SYY,*SXY,*SXZ,*SZY,*SZZ;
    float *divv,*vorv;
    float topo_scaling1, topo_scaling2;

    allocate_STD_mem(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);

    /* this one is for szz */
    compute_nodal_stress(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);

   topo_scaling1 = topo_scaling2 = 1.0;

   for(snode=1;snode<=E->lmesh.nsf;snode++)   {
      node = E->surf_node[snode];
      tpg[snode]  = -2*SZZ[node]               + SZZ[node-1];
      tpgb[snode] =  2*SZZ[node-E->lmesh.noz+1]- SZZ[node-E->lmesh.noz+2];

      tpg[snode]  =  tpg[snode] *topo_scaling1;
      tpgb[snode]  = tpgb[snode]*topo_scaling2;

      divg[snode] = 2*divv[node]-divv[node-1];
      vort[snode] = 2*vorv[node]-vorv[node-1];
   }

   free_STD_mem(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);
}

void get_STD_freesurf(struct All_variables *E,float *freesurf)
{
  int node,snode,m;

  if (E->parallel.me_loc[3]==E->parallel.nprocz-1)
    for(snode=1;snode<=E->lmesh.nsf;snode++) {
      node = E->surf_node[snode];
      freesurf[snode] += E->sphere.cap.V[3][node] * E->advection.timestep;
    }
}

void allocate_STD_mem(struct All_variables *E,
                      float* SXX, float* SYY, float* SZZ,
                      float* SXY, float* SXZ, float* SZY,
                      float* divv, float* vorv)
{
  int i;

  SXX = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
  SYY = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
  SXY = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
  SXZ = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
  SZY = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
  SZZ = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
  divv = (float *)malloc((E->lmesh.nno+1)*sizeof(float));
  vorv = (float *)malloc((E->lmesh.nno+1)*sizeof(float));

  for(i=1;i<=E->lmesh.nno;i++) {
    SZZ[i] = 0.0;
    SXX[i] = 0.0;
    SYY[i] = 0.0;
    SXY[i] = 0.0;
    SXZ[i] = 0.0;
    SZY[i] = 0.0;
    divv[i] = 0.0;
    vorv[i] = 0.0;
  }
}

void free_STD_mem(struct All_variables *E,
                  float* SXX, float* SYY, float* SZZ,
                  float* SXY, float* SXZ, float* SZY,
                  float* divv, float* vorv)
{
    free((void *)SXX);
    free((void *)SYY);
    free((void *)SXY);
    free((void *)SXZ);
    free((void *)SZY);
    free((void *)SZZ);
    free((void *)divv);
    free((void *)vorv);
}

void compute_nodal_stress(struct All_variables *E,
                          float* SXX, float* SYY, float* SZZ,
                          float* SXY, float* SXZ, float* SZY,
                          float* divv, float* vorv)
{
  void get_rtf_at_vpts();
  void velo_from_element();
  void stress_conform_bcs();
  void construct_c3x3matrix_el();
  void get_ba();

  int i,j,p,q,e,node,m,l1,l2;

  float VV[4][9],Vxyz[9][9],Szz,Sxx,Syy,Sxy,Sxz,Szy,div,vor;
  double dilation[9];
  double ba[9][9][4][7];

  double pre[9],tww[9],rtf[4][9];
  double velo_scaling, stress_scaling, mass_fac;
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
  double D[6][6],n[3],eps[6],str[6];
#endif
  struct Shape_function_dA *dOmega;
  struct Shape_function_dx *GNx;



  const int sphere_key=1;
  const int lev = E->mesh.levmax;
  const int dims = E->mesh.nsd;
  const int nel = E->lmesh.nel;
  const int vpts = vpoints[dims];
  const int ends = enodes[dims];

    for(e=1;e <= nel;e++)  {
      Szz = 0.0;
      Sxx = 0.0;
      Syy = 0.0;
      Sxy = 0.0;
      Sxz = 0.0;
      Szy = 0.0;
      div = 0.0;
      vor = 0.0;

      get_rtf_at_vpts(E, lev, e, rtf);// gets r,theta,phi coordinates at the integration points
      velo_from_element(E,VV,e,sphere_key); /* assign node-global
						 velocities to nodes
						 local to the
						 element */
      dOmega = &(E->gDA[e]);	/* Jacobian at integration points */
      GNx = &(E->gNX[e]);	/* derivatives of shape functions at
				   integration points */

      /* Vxyz is the strain rate vector, whose relationship with
       * the strain rate tensor (e) is that:
       *    Vxyz[1] = e11
       *    Vxyz[2] = e22
       *    Vxyz[3] = e33
       *    Vxyz[4] = 2*e12
       *    Vxyz[5] = 2*e13
       *    Vxyz[6] = 2*e23
       * where 1 is theta, 2 is phi, and 3 is r
       */
      for(j=1;j <= vpts;j++)  {	/* loop through velocity Gauss points  */
	/* E->EVi[j] = E->EVI[E->mesh.levmax][j]; */
        pre[j] =  E->EVi[(e-1)*vpts+j]*dOmega->vpt[j];
        dilation[j] = 0.0;
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
        for(j=1;j <= vpts;j++)	/* weighting, consisting of Jacobian,
				   Gauss weight and shape function,
				   evaluated at integration points */
          tww[i] += dOmega->vpt[j] * g_point[j].weight[E->mesh.nsd-1]
            * E->N.vpt[GNVINDEX(i,j)];
      }
      if (E->control.precise_strain_rate){
	/*  
	    B method 
	*/

	if ((e-1)%E->lmesh.elz==0) {
	  construct_c3x3matrix_el(E,e,&E->element_Cc,&E->element_Ccx,lev,0);
	}
	/* get B at velocity Gauss points */
	get_ba(&(E->N), GNx, &E->element_Cc, &E->element_Ccx,rtf, dims, ba);
	/* regular stress tensor */
	for(p=1;p <= 6;p++)
	  for(j=1;j <= vpts;j++)
	    for(i=1;i <= ends;i++)
	      for(q=1;q <= dims;q++) {
		Vxyz[p][j] += ba[i][j][q][p] * VV[q][i];
	      }
	
	/* divergence and vorticity */
	for(j=1;j <= vpts;j++)   {	/* Gauss integration points */
	  for(i=1;i <= ends;i++)   { /* nodes in element loop */
	    Vxyz[7][j]+=rtf[3][j] * (
				     VV[1][i]*GNx->vpt[GNVXINDEX(0,i,j)]
				     + VV[1][i]*E->N.vpt[GNVINDEX(i,j)]*cos(rtf[1][j])/sin(rtf[1][j])
				     + VV[2][i]*GNx->vpt[GNVXINDEX(1,i,j)]/sin(rtf[1][j])  );
	    Vxyz[8][j]+=rtf[3][j]/sin(rtf[1][j])*
	      ( VV[2][i]*GNx->vpt[GNVXINDEX(0,i,j)]*sin(rtf[1][j])
		+ VV[2][i]*E->N.vpt[GNVINDEX(i,j)]*cos(rtf[1][j])
		- VV[1][i]*GNx->vpt[GNVXINDEX(1,i,j)] );
	  }
	}

      }else{	
	/* old method */

	/* integrate over element  */
	for(j=1;j <= vpts;j++)   {	/* Gauss integration points */
	  for(i=1;i <= ends;i++)   { /* nodes in element loop */
	    /* strain rate contributions from each node */
	    Vxyz[1][j]+=( VV[1][i]*GNx->vpt[GNVXINDEX(0,i,j)]
			  + VV[3][i]*E->N.vpt[GNVINDEX(i,j)] )*rtf[3][j];
	    Vxyz[2][j]+=( (VV[2][i]*GNx->vpt[GNVXINDEX(1,i,j)]
			   + VV[1][i]*E->N.vpt[GNVINDEX(i,j)]*cos(rtf[1][j]))/sin(rtf[1][j])
			  + VV[3][i]*E->N.vpt[GNVINDEX(i,j)] )*rtf[3][j];
	    Vxyz[3][j]+= VV[3][i]*GNx->vpt[GNVXINDEX(2,i,j)];
	    
	    Vxyz[4][j]+=( (VV[1][i]*GNx->vpt[GNVXINDEX(1,i,j)]
			   - VV[2][i]*E->N.vpt[GNVINDEX(i,j)]*cos(rtf[1][j]))/sin(rtf[1][j])
			  + VV[2][i]*GNx->vpt[GNVXINDEX(0,i,j)])*rtf[3][j];
	    Vxyz[5][j]+=VV[1][i]*GNx->vpt[GNVXINDEX(2,i,j)] + rtf[3][j]*(VV[3][i]
									 *GNx->vpt[GNVXINDEX(0,i,j)]-VV[1][i]*E->N.vpt[GNVINDEX(i,j)]);
	    Vxyz[6][j]+=VV[2][i]*GNx->vpt[GNVXINDEX(2,i,j)] + rtf[3][j]*(VV[3][i]
									 *GNx->vpt[GNVXINDEX(1,i,j)]/sin(rtf[1][j])-VV[2][i]*E->N.vpt[GNVINDEX(i,j)]);
	    Vxyz[7][j]+=rtf[3][j] * (
				     VV[1][i]*GNx->vpt[GNVXINDEX(0,i,j)]
				     + VV[1][i]*E->N.vpt[GNVINDEX(i,j)]*cos(rtf[1][j])/sin(rtf[1][j])
				     + VV[2][i]*GNx->vpt[GNVXINDEX(1,i,j)]/sin(rtf[1][j])  );
	    Vxyz[8][j]+=rtf[3][j]/sin(rtf[1][j])*
	      ( VV[2][i]*GNx->vpt[GNVXINDEX(0,i,j)]*sin(rtf[1][j])
		+ VV[2][i]*E->N.vpt[GNVINDEX(i,j)]*cos(rtf[1][j])
		- VV[1][i]*GNx->vpt[GNVXINDEX(1,i,j)] );
	  }
	}
      }

      if(E->control.inv_gruneisen != 0) { /* isotropic component */
          for(j=1;j<=vpts;j++)
              dilation[j] = (Vxyz[1][j] + Vxyz[2][j] + Vxyz[3][j]) / 3.0;
      }
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
    if(E->viscosity.allow_anisotropic_viscosity){ /* general anisotropic */
      for(i=1;i <= vpts;i++)   {
	l1 = (e-1)*vpts+i;
	/* 
	   get viscosity matrix and convert to spherical system in
	   CitcomS convection

	*/
	get_constitutive(D,rtf[1][i],rtf[2][i],TRUE,
			 E->EVIn1[E->mesh.levmax][l1], 
			 E->EVIn2[E->mesh.levmax][l1], 
			 E->EVIn3[E->mesh.levmax][l1],
			 E->EVI2[E->mesh.levmax][l1],
			 E->avmode[E->mesh.levmax][l1],
			 E);
	
	/* deviatoric stress, pressure will be added later */
	eps[0] = Vxyz[1][i] - dilation[i]; /* strain-rates */
	eps[1] = Vxyz[2][i] - dilation[i];
	eps[2] = Vxyz[3][i] - dilation[i];
	eps[3] = Vxyz[4][i];
	eps[4] = Vxyz[5][i];
	eps[5] = Vxyz[6][i];
	for(l1=0;l1 < 6;l1++){	
	  str[l1]=0.0;
	  for(l2=0;l2 < 6;l2++)
	    str[l1] += D[l1][l2] * eps[l2];
	}
	Sxx += pre[i] * str[0];
	Syy += pre[i] * str[1];
	Szz += pre[i] * str[2];
	Sxy += pre[i] * str[3];
	Sxz += pre[i] * str[4];
	Szy += pre[i] * str[5];
	
	div += Vxyz[7][i]*dOmega->vpt[i]; /* divergence */
	vor += Vxyz[8][i]*dOmega->vpt[i]; /* vorticity */
      }

    }else{
#endif
      for(i=1;i <= vpts;i++)   {
	/* deviatoric stress, pressure will be added later */
          Sxx += 2.0 * pre[i] * (Vxyz[1][i] - dilation[i]); /*  */
          Syy += 2.0 * pre[i] * (Vxyz[2][i] - dilation[i]);
          Szz += 2.0 * pre[i] * (Vxyz[3][i] - dilation[i]);
          Sxy += pre[i] * Vxyz[4][i]; /*  */
          Sxz += pre[i] * Vxyz[5][i];
          Szy += pre[i] * Vxyz[6][i];
          div += Vxyz[7][i]*dOmega->vpt[i]; /* divergence */
          vor += Vxyz[8][i]*dOmega->vpt[i]; /* vorticity */
      }
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
    }
#endif
      /* normalize by volume */
      Sxx /= E->eco[e].area;
      Syy /= E->eco[e].area;
      Szz /= E->eco[e].area;
      Sxy /= E->eco[e].area;
      Sxz /= E->eco[e].area;
      Szy /= E->eco[e].area;
      div /= E->eco[e].area;
      vor /= E->eco[e].area;

      /* add the pressure term */
      Szz -= E->P[e];
      Sxx -= E->P[e];
      Syy -= E->P[e];

      for(i=1;i<=ends;i++) {
        node = E->ien[e].node[i]; /* assign to global nodes */
        SZZ[node] += tww[i] * Szz;
        SXX[node] += tww[i] * Sxx;
        SYY[node] += tww[i] * Syy;
        SXY[node] += tww[i] * Sxy;
        SXZ[node] += tww[i] * Sxz;
        SZY[node] += tww[i] * Szy;
        divv[node]+= tww[i] * div;
        vorv[node]+= tww[i] * vor;
      }

    }    /* end for el */

  (E->exchange_node_f)(E,SXX,lev);
  (E->exchange_node_f)(E,SYY,lev);
  (E->exchange_node_f)(E,SZZ,lev);
  (E->exchange_node_f)(E,SXY,lev);
  (E->exchange_node_f)(E,SXZ,lev);
  (E->exchange_node_f)(E,SZY,lev);
  (E->exchange_node_f)(E,divv,lev);
  (E->exchange_node_f)(E,vorv,lev);

  stress_scaling = velo_scaling = 1.0;

    for(node=1;node<=E->lmesh.nno;node++)   {
      mass_fac = E->Mass[node]*stress_scaling;
      SZZ[node] *= mass_fac;
      SXX[node] *= mass_fac;
      SYY[node] *= mass_fac;
      SXY[node] *= mass_fac;
      SXZ[node] *= mass_fac;
      SZY[node] *= mass_fac;
      
      mass_fac = E->Mass[node]*velo_scaling;
      vorv[node] *= mass_fac;
      divv[node] *= mass_fac;
    }

  /* assign stress to all the nodes */
    for (node=1;node<=E->lmesh.nno;node++) {
      E->gstress[(node-1)*6+1] = SXX[node];
      E->gstress[(node-1)*6+2] = SYY[node];
      E->gstress[(node-1)*6+3] = SZZ[node];
      E->gstress[(node-1)*6+4] = SXY[node];
      E->gstress[(node-1)*6+5] = SXZ[node];
      E->gstress[(node-1)*6+6] = SZY[node];
    }

  /* replace boundary stresses with boundary conditions (if specified) */
  stress_conform_bcs(E);
}

void stress_conform_bcs(struct All_variables *E)
{
  int m, i, j, k, n, d;
  const unsigned sbc_flag[4] = {0, SBX, SBY, SBZ};
  /* 
     stress tensor is sorted like so: 1: xx 2: yy 3: zz 4: xy 5: xz 6: yz 
                                         tt    pp    rr    tp    tr    pr 
  */
  const int stress_index[4][4] = { {0, 0, 0, 0}, /* traction to stress tensor conversion */
                                   {0, 1, 4, 5}, /* N-S sides,  xx xy xz */
                                   {0, 4, 2, 6}, /* E-W sides   yx yy yz */
                                   {0, 5, 6, 3} }; /* U-D sides zx zy zz */

  int noxnoz;

  noxnoz = E->lmesh.nox*E->lmesh.noz;

  if(E->control.side_sbcs) {	/* side boundary conditions */

      for(i=1; i<=E->lmesh.noy; i++)
        for(j=1; j<=E->lmesh.nox; j++)
          for(k=1; k<=E->lmesh.noz; k++) {
            n = k+(j-1)*E->lmesh.noz+(i-1)*noxnoz;
            for(d=1; d<=E->mesh.nsd; d++)
              if(E->node[n] & sbc_flag[d]) {
                if(i==1)
                  E->gstress[(n-1)*6+stress_index[d][2]] = 
                    E->sbc.SB[SIDE_WEST][d][ E->sbc.node[n] ];
                if(i==E->lmesh.noy)
                  E->gstress[(n-1)*6+stress_index[d][2]] = 
                    E->sbc.SB[SIDE_EAST][d][ E->sbc.node[n] ];
                if(j==1)
                  E->gstress[(n-1)*6+stress_index[d][1]] = 
                    E->sbc.SB[SIDE_NORTH][d][ E->sbc.node[n] ];
                if(j==E->lmesh.nox)
                  E->gstress[(n-1)*6+stress_index[d][1]] = 
                    E->sbc.SB[SIDE_SOUTH][d][ E->sbc.node[n] ];
                if(k==1)
                  E->gstress[(n-1)*6+stress_index[d][3]] = 
                    E->sbc.SB[SIDE_BOTTOM][d][ E->sbc.node[n] ];
                if(k==E->lmesh.noz)
                  E->gstress[(n-1)*6+stress_index[d][3]] = 
                    E->sbc.SB[SIDE_TOP][d][ E->sbc.node[n] ];
              }
          }

  } else {
    /* 
       no side boundary conditions
    */
    if(E->mesh.toplayerbc != 0){
      /* internal BCs are allowed */
	for(i=1; i<=E->lmesh.noy; i++)
	  for(j=1; j<=E->lmesh.nox; j++)
	    for(k=1; k<=E->lmesh.noz; k++) {
	      n = k+(j-1)*E->lmesh.noz+(i-1)*noxnoz;
	      for(d=1; d<=E->mesh.nsd; d++)
		if(E->node[n] & sbc_flag[d]) {
		  /* apply internal traction vector on horizontal surface */
		  E->gstress[(n-1)*6+stress_index[d][3]] = E->sphere.cap.VB[d][n];
		}
	    }
    }else{
      /* default */
	for(i=1; i<=E->lmesh.noy; i++)
	  for(j=1; j<=E->lmesh.nox; j++)
	    for(k=1; k<=E->lmesh.noz; k++) {
	      n = k+(j-1)*E->lmesh.noz+(i-1)*noxnoz;
	      for(d=1; d<=E->mesh.nsd; d++)
		if(E->node[n] & sbc_flag[d]) {
		  if(i==1 || i==E->lmesh.noy)
		    E->gstress[(n-1)*6+stress_index[d][2]] = E->sphere.cap.VB[d][n];
		  if(j==1 || j==E->lmesh.nox)
		    E->gstress[(n-1)*6+stress_index[d][1]] = E->sphere.cap.VB[d][n];
		  if(k==1 || k==E->lmesh.noz)
		    E->gstress[(n-1)*6+stress_index[d][3]] = E->sphere.cap.VB[d][n];
		}
	    }
    }
  }
}


/* ===================================================================
   ===================================================================  */

static void geoid_from_buoyancy(struct All_variables *E,
                                float *harm_geoid[2], float *harm_geoidb[2])
{
    /* Compute the geoid due to internal density distribution.
     *
     * geoid(ll,mm) = 4*pi*G*R*(r/R)^(ll+2)*dlayer*rho(ll,mm)/g/(2*ll+1)
     *
     * E->buoyancy needs to be converted to density (-therm_exp*ref_T/Ra/g)
     * and dimensionalized (data.density). dlayer needs to be dimensionalized.
     */

    int m,k,ll,mm,node,i,j,p,noz,snode,nxnz;
    float *TT,radius,*geoid[2],dlayer,con1,grav,scaling2,scaling,radius_m;
    float cont, conb;
    double buoy2rho;

    /* some constants */
    nxnz = E->lmesh.nox*E->lmesh.noz;
    radius_m = E->data.radius_km*1e3;

    /* scale for buoyancy */
    scaling2 = -E->data.therm_exp*E->data.ref_temperature*E->data.density
      / fabs(E->control.Atemp);
    /* scale for geoid */
    scaling = 4.0 * M_PI * 1.0e3 * E->data.radius_km * E->data.grav_const
        / E->data.grav_acc;

    /* density of one layer */
    TT = (float *) malloc ((E->lmesh.nsf+1)*sizeof(float));

    /* cos coeff */
    geoid[0] = (float*)malloc(E->sphere.hindice*sizeof(float));
    /* sin coeff */
    geoid[1] = (float*)malloc(E->sphere.hindice*sizeof(float));

    /* reset arrays */
    for (p = 0; p < E->sphere.hindice; p++) {
        harm_geoid[0][p] = 0;
        harm_geoid[1][p] = 0;
        harm_geoidb[0][p] = 0;
        harm_geoidb[1][p] = 0;
    }

    /* loop over each layer, notice the range is [1,noz) */
    for(k=1;k<E->lmesh.noz;k++)  {
        /* correction for variable gravity */
        grav = 0.5 * (E->refstate.gravity[k] + E->refstate.gravity[k+1]);
        buoy2rho = scaling2 / grav;
        for(i=1;i<=E->lmesh.noy;i++)
            for(j=1;j<=E->lmesh.nox;j++)  {
                node= k + (j-1)*E->lmesh.noz + (i-1)*nxnz;
                p = j + (i-1)*E->lmesh.nox;
                /* convert non-dimensional buoyancy to */
                /* dimensional density */
                TT[p] = (E->buoyancy[node]+E->buoyancy[node+1])*0.5*buoy2rho;
            }

        /* expand TT into spherical harmonics */
        sphere_expansion(E,TT,geoid[0],geoid[1]);

        /* thickness of the layer */
        dlayer = (E->sx[3][k+1]-E->sx[3][k])*radius_m;

        /* mean radius of the layer */
        radius = (E->sx[3][k+1]+E->sx[3][k])*0.5;

        /* geoid contribution of density at this layer, ignore degree-0 term */
        for (ll=1;ll<=E->output.llmax;ll++) {
            con1 = scaling * dlayer / (2.0*ll+1.0);
            cont = pow(radius, ((double)(ll+2)));
            conb = radius * pow(E->sphere.ri/radius, ((double)(ll)));

            for (mm=0;mm<=ll;mm++)   {
                p = E->sphere.hindex[ll][mm];
                harm_geoid[0][p] += con1*cont*geoid[0][p];
                harm_geoid[1][p] += con1*cont*geoid[1][p];
                harm_geoidb[0][p] += con1*conb*geoid[0][p];
                harm_geoidb[1][p] += con1*conb*geoid[1][p];
            }
        }

        //if(E->parallel.me==0)  fprintf(stderr,"layer %d %.5e %g %g %g\n",k,radius,dlayer,con1,con2);
    }

    /* accumulate geoid from all layers to the surface (top processors) */
    sum_across_depth_sph1(E, harm_geoid[0], harm_geoid[1]);

    /* accumulate geoid from all layers to the CMB (bottom processors) */
    sum_across_depth_sph1(E, harm_geoidb[0], harm_geoidb[1]);

    free ((void *)TT);

    free ((void *)geoid[0]);
    free ((void *)geoid[1]);
}

static void expand_topo_sph_harm(struct All_variables *E,
                                 float *tpgt[2],
                                 float *tpgb[2])
{
    /* Expand topography into spherical harmonics
     *
     * E->slice.tpg is essentailly non-dimensional stress(rr) and need
     * to be dimensionalized by stress_scaling/(delta_rho*g).
     */

    float scaling, stress_scaling, topo_scaling1,topo_scaling2;
    float den_contrast1, den_contrast2, grav1, grav2;
    int i, j;

    stress_scaling = E->data.ref_viscosity*E->data.therm_diff/
        (E->data.radius_km*E->data.radius_km*1e6);

    /* density contrast across surface, need to dimensionalize reference density */
    den_contrast1 = E->data.density*E->refstate.rho[E->lmesh.noz] - E->data.density_above;
    /* density contrast across CMB, need to dimensionalize reference density */
    den_contrast2 = E->data.density_below - E->data.density*E->refstate.rho[1];

    /* gravity at surface */
    grav1 = E->refstate.gravity[E->lmesh.noz] * E->data.grav_acc;
    /* gravity at CMB */
    grav2 = E->refstate.gravity[1] * E->data.grav_acc;

    /* scale for surface and CMB topo */
    topo_scaling1 = stress_scaling / (den_contrast1 * grav1);
    topo_scaling2 = stress_scaling / (den_contrast2 * grav2);

    /* scale for geoid */
    scaling = 4.0 * M_PI * 1.0e3 * E->data.radius_km * E->data.grav_const
        / E->data.grav_acc;

    if (E->parallel.me_loc[3] == E->parallel.nprocz-1) {
        /* expand surface topography into sph. harm. */
        sphere_expansion(E, E->slice.tpg, tpgt[0], tpgt[1]);

        /* dimensionalize surface topography */
        for (j=0; j<2; j++)
            for (i=0; i<E->sphere.hindice; i++) {
                tpgt[j][i] *= topo_scaling1;
            }
    }


    if (E->parallel.me_loc[3] == 0) {
        /* expand bottom topography into sph. harm. */
        sphere_expansion(E, E->slice.tpgb, tpgb[0], tpgb[1]);

        /* dimensionalize bottom topography */
        for (j=0; j<2; j++)
            for (i=0; i<E->sphere.hindice; i++) {
                tpgb[j][i] *= topo_scaling2;
            }
    }

    /* send arrays to all processors in the same vertical column */
    broadcast_vertical(E, tpgb[0], tpgb[1], 0);
    broadcast_vertical(E, tpgt[0], tpgt[1], E->parallel.nprocz-1);
}


static void geoid_from_topography(struct All_variables *E,
                                  float *tpgt[2],
                                  float *tpgb[2],
                                  float *geoid_tpgt[2],
                                  float *geoid_tpgb[2])
{
    /* Compute the geoid due to surface and CMB dynamic topography.
     *
     * geoid(ll,mm) = 4*pi*G*R*delta_rho*topo(ll,mm)/g/(2*ll+1)
     *
     * In theory, the degree-0 and 1 coefficients of topography must be 0.
     * The geoid coefficents for these degrees are ingnored as a result.
     */

    float con1,con2,scaling,den_contrast1,den_contrast2;
    int i,j,k,ll,mm,s;

    /* density contrast across surface, need to dimensionalize reference density */
    den_contrast1 = E->data.density*E->refstate.rho[E->lmesh.noz] - E->data.density_above;
    /* density contrast across CMB, need to dimensionalize reference density */
    den_contrast2 = E->data.density_below - E->data.density*E->refstate.rho[1];


    /* reset arrays */
    for (i = 0; i < E->sphere.hindice; i++) {
        geoid_tpgt[0][i] = 0;
        geoid_tpgt[1][i] = 0;
        geoid_tpgb[0][i] = 0;
        geoid_tpgb[1][i] = 0;
    }

    if (E->parallel.me_loc[3] == E->parallel.nprocz-1) {
        /* scale for geoid */
        scaling = 4.0 * M_PI * 1.0e3 * E->data.radius_km * E->data.grav_const
            / E->data.grav_acc;

        /* compute geoid due to surface topo, skip degree-0 and 1 term */
        for (j=0; j<2; j++)
            for (ll=2; ll<=E->output.llmax; ll++)   {
                con1 = den_contrast1 * scaling / (2.0*ll + 1.0);
                for (mm=0; mm<=ll; mm++)   {
                    i = E->sphere.hindex[ll][mm];
                    geoid_tpgt[j][i] = tpgt[j][i] * con1;
            }
        }
    }


    if (E->parallel.me_loc[3] == 0) {
        /* scale for geoid */
        scaling = 1.0e3 * 4.0 * M_PI * E->data.radius_km * E->data.grav_const
            / (E->data.grav_acc * E->refstate.gravity[1]);

        /* compute geoid due to bottom topo, skip degree-0 and 1 term */
        for (j=0; j<2; j++)
            for (ll=2; ll<=E->output.llmax; ll++)   {
                con1 = den_contrast2 * scaling / (2.0*ll + 1.0);
                con2 = con1 * pow(E->sphere.ri, ((double)(ll+2)));
                for (mm=0; mm<=ll; mm++)   {
                    i = E->sphere.hindex[ll][mm];
                    geoid_tpgb[j][i] = tpgb[j][i] * con2;
            }
        }
    }

    /* send arrays to all processors in the same vertical column */
    broadcast_vertical(E, geoid_tpgb[0], geoid_tpgb[1], 0);
    broadcast_vertical(E, geoid_tpgt[0], geoid_tpgt[1], E->parallel.nprocz-1);
}

static void geoid_from_topography_self_g(struct All_variables *E,
                                         float *tpgt[2],
                                         float *tpgb[2],
                                         float *geoid_bncy[2],
                                         float *geoid_bncy_botm[2],
                                         float *geoid_tpgt[2],
                                         float *geoid_tpgb[2])
{
    /* geoid correction due to self gravitation. The equation can be
     * found in this reference:
     * Zhong et al., (2008), A Benchmark Study on Mantle Convection
     * in a 3-D Spherical Shell Using CitcomS, submitted to G^3.
     */

    double den_contrast1,den_contrast2,grav1,grav2;
    double topo2stress1, topo2stress2;
    long double con4, ri;
    long double a1,b1,c1_0,c1_1,a2,b2,c2_0,c2_1,a11,a12,a21,a22,f1_0,f2_0,f1_1,f2_1,denom;
    int i,j,k,ll,mm,s;

    ri = E->sphere.ri;

    /* density contrast across surface, need to dimensionalize reference density */
    den_contrast1 = E->data.density*E->refstate.rho[E->lmesh.noz] - E->data.density_above;
    /* density contrast across CMB, need to dimensionalize reference density */
    den_contrast2 = E->data.density_below - E->data.density*E->refstate.rho[1];

    /* gravity at surface */
    grav1 = E->refstate.gravity[E->lmesh.noz] * E->data.grav_acc;
    /* gravity at CMB */
    grav2 = E->refstate.gravity[1] * E->data.grav_acc;

    /* scale from surface and CMB topo to stress */
    topo2stress1 = den_contrast1 * grav1;
    topo2stress2 = den_contrast2 * grav2;


    con4 = 4.0*M_PI*E->data.grav_const*E->data.radius_km*1000;

    /* reset arrays */
    for (i = 0; i < E->sphere.hindice; i++) {
        geoid_tpgt[0][i] = 0;
        geoid_tpgt[1][i] = 0;
        geoid_tpgb[0][i] = 0;
        geoid_tpgb[1][i] = 0;
    }

    for (ll=2;ll<=E->output.llmax;ll++)   {
        // dimension: gravity
        a1 = con4/(2*ll+1)*ri*lg_pow(ri,ll+1)*den_contrast2;
        b1 = con4/(2*ll+1)*den_contrast1;
        a2 = con4/(2*ll+1)*ri*den_contrast2;
        b2 = con4/(2*ll+1)*lg_pow(ri,ll)*den_contrast1;

        // dimension: rho*g
        a11 = den_contrast1*E->data.grav_acc - E->data.density*b1;
        a12 =                                - E->data.density*a1;
        a21 =-den_contrast2*b2;
        a22 = den_contrast2*(E->data.grav_acc-a2);

        denom = 1.0L / (a11*a22 - a12*a21);

        for (mm=0;mm<=ll;mm++)   {
            i = E->sphere.hindex[ll][mm];

            // cos term
            c1_0 = geoid_bncy[0][i]*E->data.density*grav1;
            c2_0 = geoid_bncy_botm[0][i]*den_contrast2*grav2;
            f1_0 = tpgt[0][i]*topo2stress1 + c1_0;
            f2_0 = tpgb[0][i]*topo2stress2 + c2_0;

            tpgt[0][i] = (f1_0*a22-f2_0*a12)*denom;
            tpgb[0][i] = (f2_0*a11-f1_0*a21)*denom;

            // sin term
            c1_1 = geoid_bncy[1][i]*E->data.density*grav1;
            c2_1 = geoid_bncy_botm[1][i]*den_contrast2*grav2;
            f1_1 = tpgt[1][i]*topo2stress1 + c1_1;
            f2_1 = tpgb[1][i]*topo2stress2 + c2_1;

            /* update topo with self-g */
            tpgt[1][i] = (f1_1*a22-f2_1*a12)*denom;
            tpgb[1][i] = (f2_1*a11-f1_1*a21)*denom;


            /* update geoid due to topo with self-g */
            geoid_tpgt[0][i] = b1 * tpgt[0][i] / grav1;
            geoid_tpgt[1][i] = b1 * tpgt[1][i] / grav1;

            geoid_tpgb[0][i] = a1 * tpgb[0][i] / grav1;
            geoid_tpgb[1][i] = a1 * tpgb[1][i] / grav1;

            /* the followings are geoid at the bottom due to topo, not used */
            //geoidb_tpg[0][i] = (a2*tpgb[0][i] +
            //                    b2*tpgt[0][i]) / grav2;

            //geoidb_tpg[1][i] = (a2*tpgb[1][i] +
            //                    b2*tpgt[1][i]) / grav2;
        }
    }

    /* send arrays to all processors in the same vertical column */
    broadcast_vertical(E, geoid_tpgb[0], geoid_tpgb[1], 0);
    broadcast_vertical(E, geoid_tpgt[0], geoid_tpgt[1], E->parallel.nprocz-1);
}

void compute_geoid( struct All_variables *E )
{
    int i, p;

    geoid_from_buoyancy(E, E->sphere.harm_geoid_from_bncy,
                        E->sphere.harm_geoid_from_bncy_botm);

    expand_topo_sph_harm(E, E->sphere.harm_tpgt, E->sphere.harm_tpgb);

    if(E->control.self_gravitation)
        geoid_from_topography_self_g(E,
                                     E->sphere.harm_tpgt,
                                     E->sphere.harm_tpgb,
                                     E->sphere.harm_geoid_from_bncy,
                                     E->sphere.harm_geoid_from_bncy_botm,
                                     E->sphere.harm_geoid_from_tpgt,
                                     E->sphere.harm_geoid_from_tpgb);
    else
        geoid_from_topography(E, E->sphere.harm_tpgt, E->sphere.harm_tpgb,
                              E->sphere.harm_geoid_from_tpgt,
                              E->sphere.harm_geoid_from_tpgb);

    if (E->parallel.me == (E->parallel.nprocz-1))  {
        for (i = 0; i < 2; i++)
            for (p = 0; p < E->sphere.hindice; p++) {
                E->sphere.harm_geoid[i][p]
                    = E->sphere.harm_geoid_from_bncy[i][p]
                    + E->sphere.harm_geoid_from_tpgt[i][p]
                    + E->sphere.harm_geoid_from_tpgb[i][p];
            }
    }
}

/* ===================================================================
   Consistent boundary flux method for stress ... Zhong,Gurnis,Hulbert

   Solve for the stress as the code defined it internally, rather than
   what was intended to be solved. This is more appropriate.

   Note also that the routine is dependent on the method
   used to solve for the velocity in the first place.
   ===================================================================  */
/* 
this routine does not require stress tensor computation, call
separately if stress output is needed
 */
void get_CBF_topo(E,H,HB)       /* call this only for top and bottom processors*/
    struct All_variables *E;
    float *H,*HB;

{
    void get_elt_k();
    void get_elt_g();
    void get_elt_f();
    void get_global_1d_shape_fn_L();
    void full_exchange_snode_f();
    void regional_exchange_snode_f();
    void velo_from_element();

    int a,address,el,elb,els,node,nodeb,nodes,i,j,k,l,m,n,count;
    int nodel,nodem,nodesl,nodesm,nnsf,nel2;

    struct Shape_function1 GM,GMb;
    struct Shape_function1_dA dGammax,dGammabx;

    float *eltTU,*eltTL,*SU,*SL,*RU,*RL;
    float VV[4][9];

    double eltk[24*24],eltf[24];
    double eltkb[24*24],eltfb[24];
    double res[24],resb[24],eu[24],eub[24];
    higher_precision eltg[24][1],eltgb[24][1];

    const int dims=E->mesh.nsd;
    const int Tsize=5;   /* maximum values, applicable to 3d, harmless for 2d */
    const int Ssize=4;
    const int ends=enodes[dims];
    const int noz=E->lmesh.noz;
    const int noy=E->lmesh.noy;
    const int nno=E->lmesh.nno;
    const int onedv=onedvpoints[dims];
    const int snode1=1,snode2=4,snode3=5,snode4=8;
    const int elz = E->lmesh.elz;
    const int ely = E->lmesh.ely;
    const int lev=E->mesh.levmax;
    const int sphere_key=1;

    const int lnsf=E->lmesh.nsf;

    eltTU = (float *)malloc((1+Tsize)*sizeof(float));
    eltTL = (float *)malloc((1+Tsize)*sizeof(float));

    SU = (float *)malloc((1+lnsf)*sizeof(float));
    SL = (float *)malloc((1+lnsf)*sizeof(float));
    RU = (float *)malloc((1+lnsf)*sizeof(float));
    RL = (float *)malloc((1+lnsf)*sizeof(float));

  for(i=0;i<=lnsf;i++)
    RU[i] = RL[i] = SU[i] = SL[i] = 0.0;

  /* calculate the element residuals */

  for(els=1;els<=E->lmesh.snel;els++) {
    el = E->surf_element[els];
    elb = el - elz+1;

    velo_from_element(E,VV,elb,sphere_key);

    for(m=0;m<ends;m++) {
       eub [m*dims  ] = VV[1][m+1];
       eub [m*dims+1] = VV[2][m+1];
       eub [m*dims+2] = VV[3][m+1];
       }

    velo_from_element(E,VV,el,sphere_key);

    for(m=0;m<ends;m++) {
       eu [m*dims  ] = VV[1][m+1];
       eu [m*dims+1] = VV[2][m+1];
       eu [m*dims+2] = VV[3][m+1];
       }

    /* The statement order is important:
       elb must be executed before el when calling get_elt_f().
       Otherwise, construct_c3x3matrix_el() would be skipped incorrectly. */
    get_elt_f(E,elb,eltfb,1);
    get_elt_f(E,el,eltf,1);

    get_elt_k(E,elb,eltkb,lev,1);
    get_elt_k(E,el,eltk,lev,1);

    if (E->control.augmented_Lagr) {
        get_aug_k(E,elb,eltkb,lev);
        get_aug_k(E,el,eltk,lev);
    }
//      get_elt_g(E,elb,eltgb,lev,j);
//      get_elt_g(E,el,eltg,lev,j);

    for(m=0;m<dims*ends;m++) {
         res[m]  = eltf[m]  - E->elt_del[lev][el].g[m][0]  * E->P[el];
         resb[m] = eltfb[m] - E->elt_del[lev][elb].g[m][0]* E->P[elb];
//           res[m]  = eltf[m] - eltg[m][0]  * E->P[el];
//           resb[m] = eltfb[m] - eltgb[m][0]* E->P[elb];
          }

    for(m=0;m<dims*ends;m++)
       for(l=0;l<dims*ends;l++) {
            res[m]  -= eltk[ends*dims*m+l]  * eu[l];
            resb[m] -= eltkb[ends*dims*m+l] * eub[l];
            }

    /* Put relevant (vertical & surface) parts of element residual into surface residual */

    for(m=1;m<=ends;m++) {
      if (m<=4)  {
        switch (m) {
           case 1:
              nodes = E->sien[els].node[1];
  break;
           case 2:
              nodes = E->sien[els].node[2];
  break;
           case 3:
              nodes = E->sien[els].node[3];
  break;
           case 4:
              nodes = E->sien[els].node[4];
  break;
     }
     RL[nodes] += resb[(m-1)*dims+2];
  }
      else   {
         switch (m) {
           case 5:
              nodes = E->sien[els].node[1];
              break;
           case 6:
              nodes = E->sien[els].node[2];
              break;
           case 7:
              nodes = E->sien[els].node[3];
              break;
           case 8:
              nodes = E->sien[els].node[4];
              break;
           }
           RU[nodes] += res[(m-1)*dims+2];
        }
      }      /* end for m */
    }


  /* calculate the LHS */

  for(els=1;els<=E->lmesh.snel;els++) {
     el = E->surf_element[els];
     elb = el - elz+1;

     get_global_1d_shape_fn_L(E,el,&GM,&dGammax,1);
     get_global_1d_shape_fn_L(E,elb,&GMb,&dGammabx,0);

     for(m=1;m<=onedv;m++)        {
        eltTU[m-1] = 0.0;
        eltTL[m-1] = 0.0;
        for(n=1;n<=onedv;n++)          {
           eltTU[m-1] +=
              dGammax.vpt[GMVGAMMA(1,n)]
              * E->L.vpt[GMVINDEX(m,n)] * E->L.vpt[GMVINDEX(m,n)];
           eltTL[m-1] +=
            dGammabx.vpt[GMVGAMMA(0,n)]
              * E->L.vpt[GMVINDEX(m,n)] * E->L.vpt[GMVINDEX(m,n)];
           }
        }

      for (m=1;m<=onedv;m++)     /* for bottom */
          SL[E->sien[els].node[m]] += eltTL[m-1];

      for (m=1;m<=onedv;m++)
          SU[E->sien[els].node[m]] += eltTU[m-1];

      }

/* for bottom topography */
if(E->parallel.me_loc[3] == 0) {
  if(E->sphere.caps == 12)
      full_exchange_snode_f(E,RL,SL,E->mesh.levmax);
  else
      regional_exchange_snode_f(E,RL,SL,E->mesh.levmax);

  for(i=1;i<=E->lmesh.nsf;i++)
      HB[i] = RL[i]/SL[i];
  }
  /* for top topo */
  if(E->parallel.me_loc[3] == E->parallel.nprocz-1) {
  if(E->sphere.caps == 12)
      full_exchange_snode_f(E,RU,SU,E->mesh.levmax);
  else
      regional_exchange_snode_f(E,RU,SU,E->mesh.levmax);

    for(i=1;i<=E->lmesh.nsf;i++)
        H[i] = RU[i]/SU[i];
  }
    free((void *)eltTU);
    free((void *)eltTL);
    free((void *)SU);
    free((void *)SL);
    free((void *)RU);
    free((void *)RL);
}


/* version */
/* $Id: Topo_gravity.c 19360 2012-01-14 08:00:41Z becker $ */

/* End of file  */
