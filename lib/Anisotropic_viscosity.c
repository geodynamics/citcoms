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



/* 

   anisotropic viscosity following Muehlhaus, Moresi, Hobbs and Dufour
   (PAGEOPH, 159, 2311, 2002)


   
*/

#ifdef CITCOM_ALLOW_ORTHOTROPIC_VISC

#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "material_properties.h"
#include "anisotropic_viscosity.h"
void calc_cbase_at_tp(float , float , float *);
void calc_cbase_at_tp_d(double , double , double *);

#define CITCOM_DELTA(i,j) ((i==j)?(1.0):(0.0))
/* 


compute a cartesian anisotropic viscosity matrix


   output: D[0,...,5][0,...,5] constitutive matrix
   input: delta_vis difference in viscosity from isotropic viscosity, which is set to unity 
   
          n[0,..,2]: director orientation, in cartesian


	  where delta_vis = (1 - eta_S/eta)


 */
void get_constitutive_orthotropic_viscosity(double D[6][6], double delta_vis,
					    double n[3], int convert_to_spherical,
					    double theta, double phi) 
{
  double nlen,delta_vis2;
  double delta[3][3][3][3],deltac[3][3][3][3];

  /* 
     make sure things are normalized (n[3] might come from interpolation)
  */
  nlen = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
  if((nlen < 0.95)||(nlen > 1.05)){
    fprintf(stderr,"get_constitutive_orthotropic_viscosity: error: nlen: %g\n",nlen);
    parallel_process_termination();
  }
  
  /* zero out most of D matrix */
            D[0][1] = D[0][2] = D[0][3] = D[0][4] = D[0][5] = 0.;
  D[1][0]           = D[1][2] = D[1][3] = D[1][4] = D[1][5] = 0.;
  D[2][0] = D[2][1]           = D[2][3] = D[2][4] = D[2][5] = 0.;
  D[3][0] = D[3][1] = D[3][2]           = D[3][4] = D[3][5] = 0.;
  D[4][0] = D[4][1] = D[4][2] = D[4][3]           = D[4][5] = 0.;
  D[5][0] = D[5][1] = D[5][2] = D[5][3] = D[5][4]           = 0.;

  /* isotropic part, in units of iso_visc */
  D[0][0] = 2.0;		/* xx = tt*/
  D[1][1] = 2.0;		/* yy = pp */
  D[2][2] = 2.0;		/* zz = rr */
  D[3][3] = 1.0;		/* xy = tp */
  D[4][4] = 1.0;		/* xz = rt */
  D[5][5] = 1.0;		/* yz = rp */


  if(fabs(delta_vis) > 5e-15){
    /* get Cartesian anisotropy matrix */
    if(convert_to_spherical){
      get_delta(deltac,n);	/* get anisotropy tensor, \Delta of
				   Muehlhaus et al. (2002)  */
      conv_cart4x4_to_spherical(deltac,theta,phi,delta); /* rotate
							    into
							    CitcomS
							    spherical
							    system  */
    }else{
      get_delta(delta,n);
    }
    delta_vis2 = 2.0*delta_vis;
    /* s_xx = s_tt */
    D[0][0] -= delta_vis2 * delta[0][0][0][0]; /* * e_xx */
    D[0][1] -= delta_vis2 * delta[0][0][1][1];
    D[0][2] -= delta_vis2 * delta[0][0][2][2];
    D[0][3] -= delta_vis  * (delta[0][0][0][1]+delta[0][0][1][0]);
    D[0][4] -= delta_vis  * (delta[0][0][0][2]+delta[0][0][2][0]);
    D[0][5] -= delta_vis  * (delta[0][0][1][2]+delta[0][0][2][1]);

    D[1][0] -= delta_vis2 * delta[1][1][0][0]; /* s_yy = s_pp */
    D[1][1] -= delta_vis2 * delta[1][1][1][1];
    D[1][2] -= delta_vis2 * delta[1][1][2][2];
    D[1][3] -= delta_vis  * (delta[1][1][0][1]+delta[1][1][1][0]);
    D[1][4] -= delta_vis  * (delta[1][1][0][2]+delta[1][1][2][0]);
    D[1][5] -= delta_vis  * (delta[1][1][1][2]+delta[1][1][2][1]);

    D[2][0] -= delta_vis2 * delta[2][2][0][0]; /* s_zz = s_rr */
    D[2][1] -= delta_vis2 * delta[2][2][1][1];
    D[2][2] -= delta_vis2 * delta[2][2][2][2];
    D[2][3] -= delta_vis  * (delta[2][2][0][1]+delta[2][2][1][0]);
    D[2][4] -= delta_vis  * (delta[2][2][0][2]+delta[2][2][2][0]);
    D[2][5] -= delta_vis  * (delta[2][2][1][2]+delta[2][2][2][1]);

    D[3][0] -= delta_vis2 * delta[0][1][0][0]; /* s_xy = s_tp */
    D[3][1] -= delta_vis2 * delta[0][1][1][1];
    D[3][2] -= delta_vis2 * delta[0][1][2][2];
    D[3][3] -= delta_vis  * (delta[0][1][0][1]+delta[0][1][1][0]);
    D[3][4] -= delta_vis  * (delta[0][1][0][2]+delta[0][1][2][0]);
    D[3][5] -= delta_vis  * (delta[0][1][1][2]+delta[0][1][2][1]);

    D[4][0] -= delta_vis2 * delta[0][2][0][0]; /* s_xz = s_tr */
    D[4][1] -= delta_vis2 * delta[0][2][1][1];
    D[4][2] -= delta_vis2 * delta[0][2][2][2];
    D[4][3] -= delta_vis  * (delta[0][2][0][1]+delta[0][2][1][0]);
    D[4][4] -= delta_vis  * (delta[0][2][0][2]+delta[0][2][2][0]);
    D[4][5] -= delta_vis  * (delta[0][2][1][2]+delta[0][2][2][1]);

    D[5][0] -= delta_vis2 * delta[1][2][0][0]; /* s_yz = s_pr */
    D[5][1] -= delta_vis2 * delta[1][2][1][1];
    D[5][2] -= delta_vis2 * delta[1][2][2][2];
    D[5][3] -= delta_vis  * (delta[1][2][0][1]+delta[1][2][1][0]);
    D[5][4] -= delta_vis  * (delta[1][2][0][2]+delta[1][2][2][0]);
    D[5][5] -= delta_vis  * (delta[1][2][1][2]+delta[1][2][2][1]);
  }
  

}

void set_anisotropic_viscosity_at_element_level(struct All_variables *E, int init_stage)
{
  int i,j,k,l,off,nel;
  double vis2,n[3],u,v,s,r;
  const int vpts = vpoints[E->mesh.nsd];
  
  if(E->viscosity.allow_orthotropic_viscosity){
    if(init_stage){	
      if(E->viscosity.orthotropic_viscosity_init)
	myerror(E,"anisotropic viscosity should not be initialized twice?!");
      /* first call */
      /* initialize anisotropic viscosity at element level, nodes will
	 get assigned later */
      switch(E->viscosity.anisotropic_init){
      case 0:			/* isotropic */
	if(E->parallel.me == 0)fprintf(stderr,"set_anisotropic_viscosity_at_element_level: initializing isotropic viscosity\n");
	for(i=E->mesh.gridmin;i <= E->mesh.gridmax;i++){
	  nel  = E->lmesh.NEL[i];
	  for (j=1;j<=E->sphere.caps_per_proc;j++) {
	    for(k=1;k <= nel;k++){
	      for(l=1;l <= vpts;l++){ /* assign to all integration points */
		off = (k-1)*vpts + l;
		E->EVI2[i][j][off] = 0.0;
		E->EVIn1[i][j][off] = 1.0; E->EVIn2[i][j][off] = E->EVIn3[i][j][off] = 0.0;
	      }
	    }
	  }
	}
	break;
      case 1:			/* 
				   random fluctuations, for testing a
				   worst case scenario

				*/
	if(E->parallel.me == 0)fprintf(stderr,"set_anisotropic_viscosity_at_element_level: initializing random viscosity\n");
	for(i=E->mesh.gridmin;i <= E->mesh.gridmax;i++){
	  nel  = E->lmesh.NEL[i];
	  for (j=1;j<=E->sphere.caps_per_proc;j++) {
	    for(k=1;k <= nel;k++){
	      /* by element (srand48 call should be in citcom.c or somewhere? */
	      vis2 = drand48()*0.9; /* random fluctuation,
				       corresponding to same strength
				       (0) and 10 fold reduction
				       (0.9) */
	      /* get random vector */
	      do{
		u = -1 + drand48()*2;v = -1 + drand48()*2;
		s = u*u + v*v;		
	      }while(s > 1);
	      r = 2.0 * sqrt(1.0-s );
	      n[0] = u * r;		/* x */
	      n[1] = v * r;		/* y */
	      n[2] = 2.0*s -1 ;		/* z */
	      for(l=1;l <= vpts;l++){ /* assign to all integration points */
		off = (k-1)*vpts + l;
		E->EVI2[i][j][off] = vis2;
		E->EVIn1[i][j][off] = n[0]; 
		E->EVIn2[i][j][off] = n[1];
		E->EVIn3[i][j][off] = n[2];
	      }
	    }
	  }
	}
	break;
      case 2:			/* from file */
#ifndef USE_GGRD	
	fprintf(stderr,"set_anisotropic_viscosity_at_element_level: anisotropic_init mode 2 requires USE_GGRD compilation\n");
	parallel_process_termination();
#endif
	ggrd_read_anivisc_from_file(E,1);
	break;
      default:
	fprintf(stderr,"set_anisotropic_viscosity_at_element_level: anisotropic_init %i undefined\n",
		E->viscosity.anisotropic_init);
	parallel_process_termination();
	break;
      }
      E->viscosity.orthotropic_viscosity_init = TRUE;
      /* end initialization stage */
    }else{
      /* standard operation every time step */


    }
  } /* end anisotropic viscosity branch */
}
#endif

void normalize_director_at_nodes(struct All_variables *E,float **n1,float **n2, float **n3, int lev)
{
  int n,m;
  double nlen;
  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(n=1;n<=E->lmesh.NNO[lev];n++){
      nlen = sqrt(n1[m][n]*n1[m][n] + n2[m][n]*n2[m][n] + n3[m][n]*n3[m][n]);
      n1[m][n] /= nlen;
      n2[m][n] /= nlen;
      n3[m][n] /= nlen;
    }
}
void normalize_director_at_gint(struct All_variables *E,float **n1,float **n2, float **n3, int lev)
{
  int m,e,i,off;
  const int nsd=E->mesh.nsd;
  const int vpts=vpoints[nsd];
  double nlen;
  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(e=1;e<=E->lmesh.NEL[lev];e++)
      for(i=1;i<=vpts;i++)      {
	off = (e-1)*vpts+i;
	nlen = sqrt(n1[m][off]*n1[m][off] + n2[m][off]*n2[m][off] + n3[m][off]*n3[m][off]);
	n1[m][off] /= nlen;
	n2[m][off] /= nlen;
	n3[m][off] /= nlen;
      }
}
/* 

convert cartesian voigt matrix to spherical, CitcomS format

1: t 2: p 3: r

(E only passed for debugging)

*/

  void conv_cart4x4_to_spherical(double c[3][3][3][3], double theta, double phi, double p[3][3][3][3])
{
  double rot[3][3],base[9];
  calc_cbase_at_tp_d(theta,phi, base); /* compute cartesian basis at
					  theta, phi location */
  rot[0][0] = base[3];rot[0][1] = base[4];rot[0][2] = base[5]; /* theta */
  rot[1][0] = base[6];rot[1][1] = base[7];rot[1][2] = base[8]; /* phi */
  rot[2][0] = base[0];rot[2][1] = base[1];rot[2][2] = base[2]; /* r */
  //fprintf(stderr,"%g %g ; %g %g %g ; %g %g %g ; %g %g %g\n\n",theta,phi,rot[0][0],rot[0][1],rot[0][2],rot[1][0],rot[1][1],rot[1][2],rot[2][0],rot[2][1],rot[2][2]);
  rot_4x4(c,rot,p);
  //if(E->parallel.me==0)print_6x6_mat(stderr,p);
  //if(E->parallel.me==0)fprintf(stderr,"\n\n");
}

/* 


get anisotropy matrix following Muehlhaus

 */
void get_delta(double d[3][3][3][3],double n[3])
{
  int i,j,k,l;
  double tmp;
  for(i=0;i<3;i++)
    for(j=0;j<3;j++)
      for(k=0;k<3;k++)
	for(l=0;l<3;l++){	/* eq. (4) from Muehlhaus et al. (2002) */
	  tmp  = n[i]*n[k]*CITCOM_DELTA(l,j);
	  tmp += n[j]*n[k]*CITCOM_DELTA(i,l);
	  tmp += n[i]*n[l]*CITCOM_DELTA(k,j);
	  tmp += n[j]*n[l]*CITCOM_DELTA(i,k);
	  tmp /= 2.0;
	  tmp -= 2*n[i]*n[j]*n[k]*n[l];
	  d[i][j][k][l] = tmp;
	}

}


/* rotate fourth order tensor */
void rot_4x4(double c4[3][3][3][3], double r[3][3], double c4c[3][3][3][3])
{

  int i1,i2,i3,i4,j1,j2,j3,j4;
  for(i1=0;i1<3;i1++)  
    for(i2=0;i2<3;i2++)  
      for(i3=0;i3<3;i3++) 
	for(i4=0;i4<3;i4++) 
	  c4c[i1][i2][i3][i4] = 0.0;
  
  for(i1=0;i1<3;i1++)
    for(i2=0;i2<3;i2++)
      for(i3=0;i3<3;i3++)
	for(i4=0;i4<3;i4++)
	  for(j1=0;j1<3;j1++)
	    for(j2=0;j2<3;j2++)
	      for(j3=0;j3<3;j3++)
		for(j4=0;j4<3;j4++)
		  c4c[i1][i2][i3][i4] += r[i1][j1] * r[i2][j2]* r[i3][j3]* r[i4][j4] * c4[j1][j2][j3][j4];

}

void print_6x6_mat(FILE *out, double c[6][6])
{
  int i,j;
  for(i=0;i<6;i++){
    for(j=0;j<6;j++)
      fprintf(out,"%10g ",(fabs(c[i][j])<5e-15)?(0):(c[i][j]));
    fprintf(out,"\n");
  }
}

