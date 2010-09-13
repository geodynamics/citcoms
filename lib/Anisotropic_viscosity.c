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

   tranverse isotropy following Han and Wahr (PEPI, 102, 33, 1997)

   
*/

#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC

#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "material_properties.h"
#include "anisotropic_viscosity.h"
void calc_cbase_at_tp(float , float , float *);
void calc_cbase_at_tp_d(double , double , double *);

#define CITCOM_DELTA(i,j) ((i==j)?(1.0):(0.0))


/* 

transversely isotropic viscosity following Han and Wahr


\nu_1 = isotropic viscosity, applies for  e_31, e_23
\nu_2 = weak anisotropy, applies for e_31, e_32
\eta_1 = normal viscosity, (\eta_1+2\nu_1) control e_11, e_22
\eta_2 = normal viscosity, (\eta_2+2\nu_2) = 2\eta_1 + 2\nu_1, controls e_33

we use (for consistency with anisotropic viscosity)

Delta = 1-\nu_2/\nu_1

and 

\Gamma, such that \eta_1 = \Gamma \nu_1

\nu_1 is the reference, isotropic viscosity, set to unity here, i.e.

\nu_2 = 1 - \Delta ; \eta_1 = \Gamma ; (\eta_2 = 2 (\Gamma-\Delta)); for isotropy \Delta = 0, \Gamma = 0

n[3] is the cartesian direction into which the weak shear points
(ie. routine will rotate the 3 axis into the n direction) and will 
normalize n, if not already normalized


*/
void get_constitutive_ti_viscosity(double D[6][6], double delta_vis, double gamma_vis,
				   double n[3], int convert_to_spherical,
				   double theta, double phi) 
{
  double nlen,delta_vis2;
  int ani;
  /* isotropic part, in units of iso_visc */
  zero_6x6(D);
  D[0][0] = 2.0;		/* xx = tt*/
  D[1][1] = 2.0;		/* yy = pp */
  D[2][2] = 2.0;		/* zz = rr */
  D[3][3] = 1.0;		/* xy = tp */
  D[4][4] = 1.0;		/* xz = rt */
  D[5][5] = 1.0;		/* yz = rp */

  ani = FALSE;
  if((fabs(delta_vis) > 3e-15) || (fabs(gamma_vis) > 3e-15)){
    ani = TRUE;
    /* get Cartesian anisotropy matrix by adding anisotropic
       components */
    D[0][0] += gamma_vis;
    D[1][0] = D[0][1] = gamma_vis;
    D[1][1] = D[0][0];
    D[2][2] += 2.*gamma_vis;
    D[4][4] -= delta_vis;
    D[5][5] = D[4][4];
    /* 
       the rotation routine takes care of normalization and will normalize n
    */
    //print_6x6_mat(stderr,D);
    rotate_ti6x6_to_director(D,n); /* rotate such that the generic z
				      preferred axis is aligned with
				      the director */
    //print_6x6_mat(stderr,D);fprintf(stderr,"\n");
  }
  if(ani && convert_to_spherical){
    conv_cart6x6_to_spherical(D,theta,phi,D); /* rotate, can use in place */
  }
}

/* 


compute a cartesian orthotropic anisotropic viscosity matrix (and
rotate it into CitcomS spherical, if requested)

viscosity is characterized by a eta_S (weak) viscosity in a shear
plane, to which the director is normal


   output: D[0,...,5][0,...,5] constitutive matrix

   input: delta_vis difference in viscosity from isotropic viscosity (set to unity here)
   
          n[0,..,2]: director orientation, specify in cartesian


	  where delta_vis = (1 - eta_S/eta)


 */
void get_constitutive_orthotropic_viscosity(double D[6][6], double delta_vis,
					    double n[3], int convert_to_spherical,
					    double theta, double phi) 
{
  double nlen,delta_vis2;
  double delta[3][3][3][3];
  int ani;
  ani=FALSE;
  /* isotropic part, in units of iso_visc */
  zero_6x6(D);
  D[0][0] = 2.0;		/* xx = tt*/
  D[1][1] = 2.0;		/* yy = pp */
  D[2][2] = 2.0;		/* zz = rr */
  D[3][3] = 1.0;		/* xy = tp */
  D[4][4] = 1.0;		/* xz = rt */
  D[5][5] = 1.0;		/* yz = rp */

  /* get Cartesian anisotropy matrix */
  if(fabs(delta_vis) > 3e-15){
    ani = TRUE;
    get_orth_delta(delta,n);	/* 
				   get anisotropy tensor, \Delta of
				   Muehlhaus et al. (2002)

				   this routine will normalize n, just in case

				*/
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
  if(ani && convert_to_spherical){
    //print_6x6_mat(stderr,D);
    conv_cart6x6_to_spherical(D,theta,phi,D); /* rotate, can use same mat for 6x6 */
    //print_6x6_mat(stderr,D);fprintf(stderr,"\n");
  }
}

void set_anisotropic_viscosity_at_element_level(struct All_variables *E, int init_stage)
{
  int i,j,k,l,off,nel;
  double vis2,n[3],u,v,s,r;
  const int vpts = vpoints[E->mesh.nsd];
  
  if(E->viscosity.allow_anisotropic_viscosity){
    if(init_stage){	
      if(E->viscosity.anisotropic_viscosity_init)
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
      E->viscosity.anisotropic_viscosity_init = TRUE;
      /* end initialization stage */
    }else{
      /* standard operation every time step */


    }
  } /* end anisotropic viscosity branch */
}


void normalize_director_at_nodes(struct All_variables *E,float **n1,float **n2, float **n3, int lev)
{
  int n,m;
  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(n=1;n<=E->lmesh.NNO[lev];n++){
      normalize_vec3(&(n1[m][n]),&(n2[m][n]),&(n3[m][n]));
    }
}
void normalize_director_at_gint(struct All_variables *E,float **n1,float **n2, float **n3, int lev)
{
  int m,e,i,off;
  const int nsd=E->mesh.nsd;
  const int vpts=vpoints[nsd];
  for (m=1;m<=E->sphere.caps_per_proc;m++)
    for(e=1;e<=E->lmesh.NEL[lev];e++)
      for(i=1;i<=vpts;i++)      {
	off = (e-1)*vpts+i;
	normalize_vec3(&(n1[m][off]),&(n2[m][off]),&( n3[m][off]));
      }
}
/* 
   
convert cartesian fourth order tensor (input c) to spherical, CitcomS
format (output p)

c and p cannot be the same matrix

1: t 2: p 3: r

(E only passed for debugging)

*/

  void conv_cart4x4_to_spherical(double c[3][3][3][3], double theta, double phi, double p[3][3][3][3])
{
  double rot[3][3];
  get_citcom_spherical_rot(theta,phi,rot);
  rot_4x4(c,rot,p);
}

/* convert [6][6] (input c) in cartesian to citcom spherical (output
   p)

   c and p can be the same amtrix

*/
void conv_cart6x6_to_spherical(double c[6][6], double theta, double phi, double p[6][6])
{
  double c4[3][3][3][3],p4[3][3][3][3],rot[3][3];
  get_citcom_spherical_rot(theta,phi,rot);
  c4fromc6(c4,c);		
  rot_4x4(c4,rot,p4);
  c6fromc4(p,p4);
}
/* 

rotate 6x6 D matrix with preferred axis aligned with z to the
Cartesian director orientation, in place

n will be normalized, just in case

*/
void rotate_ti6x6_to_director(double D[6][6],double n[3])
{
  double a[3][3][3][3],b[3][3][3][3],rot[3][3],
    hlen2,x2,y2,xy,zm1;
  /* normalize */
  normalize_vec3d((n+0),(n+1),(n+2));
  /* calc aux variable */
  x2 = n[0]*n[0];y2 = n[1]*n[1];xy = n[0]*n[1];
  hlen2 = x2 + y2;zm1 = n[2]-1;
  /* rotation matrix to get {0,0,1} to {x,y,z} */
  rot[0][0] = (y2 + x2*n[2])/hlen2;
  rot[0][1] = (xy*zm1)/hlen2;
  rot[0][2] = n[0];
  rot[1][0] = rot[0][1];
  rot[1][1] = (x2 + y2*n[2])/hlen2;
  rot[1][2] = n[1];
  rot[2][0] = -n[0];
  rot[2][1] = -n[1];
  rot[2][2] =  n[2];
  /* rotate the D matrix */
  c4fromc6(a,D);
  rot_4x4(a,rot,b);
  c6fromc4(D,b);
}

void get_citcom_spherical_rot(double theta, double phi, double rot[3][3]){
  double base[9];
  calc_cbase_at_tp_d(theta,phi, base); /* compute cartesian basis at
					  theta, phi location */
  rot[0][0] = base[3];rot[0][1] = base[4];rot[0][2] = base[5]; /* theta */
  rot[1][0] = base[6];rot[1][1] = base[7];rot[1][2] = base[8]; /* phi */
  rot[2][0] = base[0];rot[2][1] = base[1];rot[2][2] = base[2]; /* r */
  //fprintf(stderr,"%g %g ; %g %g %g ; %g %g %g ; %g %g %g\n\n",
  //theta,phi,rot[0][0],rot[0][1],rot[0][2],rot[1][0],rot[1][1],rot[1][2],rot[2][0],rot[2][1],rot[2][2]);
}
/* 


get fourth order anisotropy tensor for orthotropic viscosity from
Muehlhaus et al. (2002)

*/
void get_orth_delta(double d[3][3][3][3],double n[3])
{
  int i,j,k,l;
  double tmp;
  normalize_vec3d((n+0),(n+1),(n+2));
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


/* 
   rotate fourth order tensor 
   c4 and c4c cannot be the same matrix

*/
void rot_4x4(double c4[3][3][3][3], double r[3][3], double c4c[3][3][3][3])
{

  int i1,i2,i3,i4,j1,j2,j3,j4;

  zero_4x4(c4c);

  for(i1=0;i1<3;i1++)
    for(i2=0;i2<3;i2++)
      for(i3=0;i3<3;i3++)
	for(i4=0;i4<3;i4++)
	  for(j1=0;j1<3;j1++)
	    for(j2=0;j2<3;j2++)
	      for(j3=0;j3<3;j3++)
		for(j4=0;j4<3;j4++)
		  c4c[i1][i2][i3][i4] += r[i1][j1] * r[i2][j2] * 
		                         r[i3][j3]* r[i4][j4]  * c4[j1][j2][j3][j4];

}
void zero_6x6(double a[6][6])
{
  int i,j;
  for(i=0;i<6;i++)
    for(j=0;j<6;j++)
      a[i][j] = 0.;
  
}
void zero_4x4(double a[3][3][3][3])
{
  int i1,i2,i3,i4;
  for(i1=0;i1<3;i1++)  
    for(i2=0;i2<3;i2++)  
      for(i3=0;i3<3;i3++) 
	for(i4=0;i4<3;i4++) 
	  a[i1][i2][i3][i4] = 0.0;
  
}
void copy_4x4(double a[3][3][3][3], double b[3][3][3][3])
{

  int i1,i2,i3,i4;
  for(i1=0;i1<3;i1++)  
    for(i2=0;i2<3;i2++)  
      for(i3=0;i3<3;i3++) 
	for(i4=0;i4<3;i4++) 
	  b[i1][i2][i3][i4] = a[i1][i2][i3][i4];
}
void copy_6x6(double a[6][6], double b[6][6])
{

  int i1,i2;
  for(i1=0;i1<6;i1++)  
    for(i2=0;i2<6;i2++)  
      b[i1][i2] = a[i1][i2];
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
/* 
   create a fourth order tensor representation from the voigt
   notation, assuming only upper half is filled in

 */
void c4fromc6(double c4[3][3][3][3],double c[6][6])
{
  int i,j;
  
  c4[0][0][0][0] =                  c[0][0];
  c4[0][0][1][1] =                  c[0][1];
  c4[0][0][2][2] =                  c[0][2];
  c4[0][0][0][1] = c4[0][0][1][0] = c[0][3];
  c4[0][0][0][2] = c4[0][0][2][0] = c[0][4];
  c4[0][0][1][2] = c4[0][0][2][1] = c[0][5];

  c4[1][1][0][0] =                  c[0][1];
  c4[1][1][1][1] =                  c[1][1];
  c4[1][1][2][2] =                  c[1][2];
  c4[1][1][0][1] = c4[1][1][1][0] = c[1][3];
  c4[1][1][0][2] = c4[1][1][2][0] = c[1][4];
  c4[1][1][1][2] = c4[1][1][2][1] = c[1][5];
 
  c4[2][2][0][0] =                  c[0][2];
  c4[2][2][1][1] =                  c[1][2];
  c4[2][2][2][2] =                  c[2][2];
  c4[2][2][0][1] = c4[2][2][1][0] = c[2][3];
  c4[2][2][0][2] = c4[2][2][2][0] = c[2][4];
  c4[2][2][1][2] = c4[2][2][2][1] = c[2][5];

  c4[0][1][0][0] =                  c[0][3];
  c4[0][1][1][1] =                  c[1][3];
  c4[0][1][2][2] =                  c[2][3];
  c4[0][1][0][1] = c4[0][1][1][0] = c[3][3];
  c4[0][1][0][2] = c4[0][1][2][0] = c[3][4];
  c4[0][1][1][2] = c4[0][1][2][1] = c[3][5];

  c4[0][2][0][0] =                  c[0][4];
  c4[0][2][1][1] =                  c[1][4];
  c4[0][2][2][2] =                  c[2][4];
  c4[0][2][0][1] = c4[0][2][1][0] = c[3][4];
  c4[0][2][0][2] = c4[0][2][2][0] = c[4][4];
  c4[0][2][1][2] = c4[0][2][2][1] = c[4][5];

  c4[1][2][0][0] =                  c[0][5];
  c4[1][2][1][1] =                  c[1][5];
  c4[1][2][2][2] =                  c[2][5];
  c4[1][2][0][1] = c4[1][2][1][0] = c[3][5];
  c4[1][2][0][2] = c4[1][2][2][0] = c[4][5];
  c4[1][2][1][2] = c4[1][2][2][1] = c[5][5];

  /* assign the symmetric diagonal terms */
  for(i=0;i<3;i++)
    for(j=0;j<3;j++){
      c4[1][0][i][j] = c4[0][1][i][j];
      c4[2][0][i][j] = c4[0][2][i][j];
      c4[2][1][i][j] = c4[1][2][i][j];
    }

}
void c6fromc4(double c[6][6],double c4[3][3][3][3])
{
  int i,j;
  
  c[0][0] = c4[0][0][0][0];
  c[0][1] = c4[0][0][1][1];
  c[0][2] = c4[0][0][2][2];
  c[0][3] = c4[0][0][0][1];
  c[0][4] = c4[0][0][0][2];
  c[0][5] = c4[0][0][1][2];

  c[1][1] = c4[1][1][1][1];
  c[1][2] = c4[1][1][2][2];
  c[1][3] = c4[1][1][0][1];
  c[1][4] = c4[1][1][0][2];
  c[1][5] = c4[1][1][1][2];

  c[2][2] = c4[2][2][2][2];
  c[2][3] = c4[2][2][0][1];
  c[2][4] = c4[2][2][0][2];
  c[2][5] = c4[2][2][1][2];
  
  c[3][3] = c4[0][1][0][1];
  c[3][4] = c4[0][1][0][2];
  c[3][5] = c4[0][1][1][2];
  
  c[4][4] = c4[0][2][0][2];
  c[4][5] = c4[0][2][1][2];
  
  c[5][5] = c4[1][2][1][2];
  /* fill in the lower half */
  for(i=0;i<6;i++)
    for(j=i+1;j<6;j++)
      c[j][i] = c[i][j];
}


void normalize_vec3(float *x, float *y, float *z)
{
  double len = 0.;
  len += (double)(*x) * (double)(*x);
  len += (double)(*y) * (double)(*y);
  len += (double)(*z) * (double)(*z);
  len = sqrt(len);
  *x /= len;*y /= len;*z /= len;
}
void normalize_vec3d(double *x, double *y, double *z)
{
  double len = 0.;
  len += (*x) * (*x);len += (*y) * (*y);len += (*z) * (*z);
  len = sqrt(len);
  *x /= len;*y /= len;*z /= len;
}


#endif
