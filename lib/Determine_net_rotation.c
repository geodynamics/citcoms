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

routines to determine the net rotation velocity of the whole model

TWB

These have been superceded by the routines in Global_opertations and can probably be removed 


*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "parallel_related.h"
#include "parsing.h"
#include "output.h"

#ifdef USE_GGRD
void hc_ludcmp_3x3(HC_PREC [3][3],int,int *);
void hc_lubksb_3x3(HC_PREC [3][3],int,int *,HC_PREC *);
#endif

double determine_netr_tp(float, float, float, float, float, int, double *, double *);
void sub_netr(float, float, float, float *, float *, double *);
void xyz2rtp(float ,float ,float ,float *);
void *safe_malloc (size_t );
double determine_model_net_rotation(struct All_variables *,double *);
void myerror(struct All_variables *,char *);
/*

determine the mean net rotation of the velocities at all layers


modeled after horizontal layer average routines

*/
double determine_model_net_rotation(struct All_variables *E,double *omega)
{
  const int dims = E->mesh.nsd;
  int m,i,j,k,d,nint,el,elz,elx,ely,j1,j2,i1,i2,k1,k2,nproc;
  int top,lnode[5],elz9;
  double *acoef,*coef,*lomega,ddummy,oamp,lamp;
  float v[2],vw,vtmp,r,t,p,x[3],xp[3],r1,r2,rr;
  struct Shape_function1 M;
  struct Shape_function1_dA dGamma;
  void get_global_1d_shape_fn();

  elz = E->lmesh.elz;elx = E->lmesh.elx;ely = E->lmesh.ely;

  elz9 = elz*9;

  acoef =  (double *)safe_malloc(elz9*sizeof(double));
  coef =   (double *)safe_malloc(elz9*sizeof(double));
  lomega = (double *)safe_malloc(3*elz*sizeof(double));

  for (i=1;i <= elz;i++)  {	/* loop through depths */

    /* zero out coef for init */
    determine_netr_tp(ddummy,ddummy,ddummy,ddummy,ddummy,0,(coef+(i-1)*9),&ddummy);

    if (i==elz)
      top = 1;
    else
      top = 0;
      for (k=1;k <= ely;k++)
        for (j=1;j <= elx;j++)     {
          el = i + (j-1)*elz + (k-1)*elx*elz;
          get_global_1d_shape_fn(E,el,&M,&dGamma,top);

	  /* find mean element location and horizontal velocity */

	  x[0] = x[1] = x[2] = v[0] = v[1] = vw = 0.0;

          lnode[1] = E->ien[el].node[1];
          lnode[2] = E->ien[el].node[2];
          lnode[3] = E->ien[el].node[3];
          lnode[4] = E->ien[el].node[4];

          for(nint=1;nint <= onedvpoints[E->mesh.nsd];nint++)   {
            for(d=1;d <= onedvpoints[E->mesh.nsd];d++){
	      vtmp = E->M.vpt[GMVINDEX(d,nint)] * dGamma.vpt[GMVGAMMA(0,nint)];
	      x[0] += E->x[1][lnode[d]] * vtmp; /* coords */
	      x[1] += E->x[2][lnode[d]] * vtmp;
	      x[2] += E->x[3][lnode[d]] * vtmp;
	      
              v[0] += E->sphere.cap[CPPR].V[1][lnode[d]] * vtmp; /* theta */
              v[1] += E->sphere.cap[CPPR].V[2][lnode[d]] * vtmp; /* phi */
	      vw += dGamma.vpt[GMVGAMMA(0,nint)];
	    }
	  }
          if (i==elz)  {
            lnode[1] = E->ien[el].node[5];
            lnode[2] = E->ien[el].node[6];
            lnode[3] = E->ien[el].node[7];
            lnode[4] = E->ien[el].node[8];

            for(nint=1;nint<=onedvpoints[E->mesh.nsd];nint++)   {
              for(d=1;d<=onedvpoints[E->mesh.nsd];d++){
		vtmp = E->M.vpt[GMVINDEX(d,nint)] * dGamma.vpt[GMVGAMMA(1,nint)];
		x[0] += E->x[1][lnode[d]] * vtmp; /* coords */
		x[1] += E->x[2][lnode[d]] * vtmp;
		x[2] += E->x[3][lnode[d]] * vtmp;
		/*  */
		v[0] += E->sphere.cap[CPPR].V[1][lnode[d]] * vtmp;
		v[1] += E->sphere.cap[CPPR].V[2][lnode[d]] * vtmp;
		vw += dGamma.vpt[GMVGAMMA(1,nint)];
	      }
	    }
	  }   /* end of if i==elz    */
	  x[0] /= vw;x[1] /= vw;x[2] /= vw; /* convert */
	  xyz2rtp(x[0],x[1],x[2],xp);
	  v[0] /= vw;v[1] /= vw;
	  /* add */
	  determine_netr_tp(xp[0],xp[1],xp[2],v[0],v[1],1,(coef+(i-1)*9),&ddummy);
	}  /* end of j  and k, and m  */

  }        /* Done for i */
  /*
     sum it all up
  */
  MPI_Allreduce(coef,acoef,elz9,MPI_DOUBLE,MPI_SUM,E->parallel.horizontal_comm);

  omega[0]=omega[1]=omega[2]=0.0;

  /* depth range */
  rr = E->sx[3][E->ien[elz].node[5]] - E->sx[3][E->ien[1].node[1]];
  if(rr < 1e-7)
    myerror(E,"rr error in net r determine");
  vw = 0.0;
  for (i=0;i < elz;i++) {	/* regular 0..n-1 loop */
    /* solve layer NR */
    lamp = determine_netr_tp(ddummy,ddummy,ddummy,ddummy,ddummy,2,(acoef+i*9),(lomega+i*3));
    r1 = E->sx[3][E->ien[i+1].node[1]]; /* nodal radii for the
						 i-th element, this
						 assumes that there
						 are no lateral
						 variations in radii!
					      */
    r2 = E->sx[3][E->ien[i+1].node[5]];
    vtmp = (r2-r1)/rr;		/* weight for this layer */
    //if(E->parallel.me == 0)
    //  fprintf(stderr,"NR layer %5i (%11g - %11g, %11g): |%11g %11g %11g| = %11g\n",
    //	      i+1,r1,r2,vtmp,lomega[i*3+0],lomega[i*3+1],lomega[i*3+2],lamp);
    /*  */
    for(i1=0;i1<3;i1++)
      omega[i1] += lomega[i*3+i1] * vtmp;
    vw += vtmp;
  }
  if(fabs(vw) > 1e-8)		/* when would it be zero? */
    for(i1=0;i1 < 3;i1++)
      omega[i1] /= vw;
  else
    for(i1=0;i1 < 3;i1++)
      omega[i1] = 0.0;
  free ((void *) acoef);
  free ((void *) coef);
  free ((void *) lomega);


  oamp = sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2]);
  if(E->parallel.me == 0)
    fprintf(stderr,"determined net rotation of | %.4e %.4e %.4e | = %.4e\n",
	    omega[0],omega[1],omega[2],oamp);
  return oamp;
}

/*



compute net rotation from velocities given at r, theta, phi as vel_theta and vel_phi

the routines below are based on code originally from Peter Bird, see
copyright notice below



the routine will only properly work for global models if the sampling
is roughly equal area!

mode: 0 initialize
      1 sum
      2 solve

*/


double determine_netr_tp(float r,float theta,float phi,
			 float velt,float velp,int mode,
			 double *c9,double *omega)
{
  float coslat,coslon,sinlat,sinlon,rx,ry,rz,rate,rzu,a,b,c,d,e,f;
  int i,j,ind[3];
  static int n3 = 3;
  double amp,coef[3][3];
  switch(mode){
  case 0:			/* initialize */
    for(i=0;i < 9;i++)
      c9[i] = 0.0;
    amp = 0.0;
    break;
  case 1:			/* add this velocity */
    if((fabs(theta) > 1e-5) &&(fabs(theta-M_PI) > 1e-5)){
      coslat=sin(theta);
      coslon=cos(phi);
      sinlat=cos(theta);
      sinlon=sin(phi);

      rx=coslat*coslon*r;
      ry=coslat*sinlon*r;
      rz=sinlat*r;

      rzu=sinlat;

      a = -rz*rzu*sinlon-ry*coslat;
      b = -rz*coslon;
      c =  rz*rzu*coslon+rx*coslat;
      d = -rz*sinlon;
      e = -ry*rzu*coslon+rx*rzu*sinlon;      

      f =  ry*sinlon+rx*coslon;

      c9[0] += a*a+b*b;
      c9[1] += a*c+b*d;
      c9[2] += a*e+b*f;
      c9[3] += c*c+d*d;
      c9[4] += c*e+d*f;
      c9[5] += e*e+f*f;
      
      c9[6] += a*velt+b*velp;
      c9[7] += c*velt+d*velp;
      c9[8] += e*velt+f*velp;
    }
    amp = 0;
    break;
  case 2:			/* solve */
    coef[0][0] = c9[0];		/* assemble matrix */
    coef[0][1] = c9[1];
    coef[0][2] = c9[2];
    coef[1][1] = c9[3];
    coef[1][2] = c9[4];
    coef[2][2] = c9[5];
    coef[1][0]=coef[0][1];	/* symmetric */
    coef[2][0]=coef[0][2];
    coef[2][1]=coef[1][2];
    /*  */
    omega[0] = c9[6];
    omega[1] = c9[7];
    omega[2] = c9[8];

    /* solve */
#ifdef USE_GGRD
    hc_ludcmp_3x3(coef,n3,ind);
    hc_lubksb_3x3(coef,n3,ind,omega);
#endif
    amp = sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2]);
    break;
  default:
    fprintf(stderr,"determine_netr_tp: mode %i undefined\n",mode);
    parallel_process_termination();
    break;
  }
  return amp;
}

//
//     subtract a net rotation component from a velocity
//     field given as v_theta (velt) and v_phi (velp)
//

void sub_netr(float r,float theta,float phi,float *velt,float *velp, double *omega)
{

  float coslat,coslon,sinlon,sinlat,rx,ry,rz;
  float vx,vy,vz,tx,ty,tz,vtheta,pc,px,py,vphi;

  coslat=sin(theta);
  coslon=cos(phi);
  sinlat=cos(theta);
  sinlon=sin(phi);

  rx = coslat*coslon*r;		/* location vector in Cartesian */
  ry = coslat*sinlon*r;
  rz = sinlat*r;

  vx = omega[1]*rz - omega[2]*ry; /* cross product */
  vy = omega[2]*rx - omega[0]*rz;
  vz = omega[0]*ry - omega[1]*rx;

  tx =  sinlat*coslon;		/* theta basis vectors */
  ty =  sinlat*sinlon;
  tz = -coslat;

  vtheta = vx*tx + vy*ty + vz*tz;

  px = -sinlon;			/* phi basis vectors */
  py = coslon;

  vphi = vx * px + vy * py;

  /* remove */
  *velt = *velt - vtheta;
  *velp = *velp - vphi;
}



//
//      PROGRAM -OrbScore-: COMPARES OUTPUT FROM -SHELLS-
//                          WITH DATA FROM GEODETI// NETWORKS,
//                          STRESS DIRECTIONS, FAULT SLIP RATES,
//                          SEAFLOOR SPREADING RATES, AND SEISMICITY,
//                          AND REPORTS SUMMARY SCALAR SCORES.
//
//=========== PART OF THE "SHELLS" PACKAGE OF PROGRAMS===========
//
//   GIVEN A FINITE ELEMENT GRID FILE, IN THE FORMAT PRODUCED BY
//  -OrbWeave- AND RENUMBERED BY -OrbNumbr-, WITH NODAL DATA
//   ADDED BY -OrbData-, AND NODE-VELOCITY OUTPUT FROM -SHELLS-,
//   COMPUTES A VARIETY OF SCORES OF THE RESULTS.
//
//   NOTE: Does not contain VISCOS or DIAMND, hence independent
//         of changes made in May 1998, and equally compatible
//         with Old_SHELLS or with improved SHELLS.
//
//                             by
//                        Peter Bird
//          Department of Earth and Spcae Sciences,
//    University of California, Los Angeles, California 90095-1567
//   (C) Copyright 1994, 1998, 1999, 2000
//                by Peter Bird and the Regents of
//                 the University of California.
//              (For version data see FORMAT 1 below)
//
//   THIS PROGRAM WAS DEVELOPED WITH SUPPORT FROM THE UNIVERSITY OF
//     CALIFORNIA, THE UNITED STATES GEOLOGI// SURVEY, THE NATIONAL
//     SCIENCE FOUNDATION, AND THE NATIONAL AERONAUTICS AND SPACE
//     ADMINISTRATION.
//   IT IS FREEWARE, AND MAY BE COPIED AND USED WITHOUT CHARGE.
//   IT MAY NOT BE MODIFIED IN A WAY WHICH HIDES ITS ORIGIN
//     OR REMOVES THIS MESSAGE OR THE COPYRIGHT MESSAGE.
//   IT MAY NOT BE RESOLD FOR MORE THAN THE COST OF REPRODUCTION
//      AND MAILING.
//

#ifdef USE_GGRD
#define NR_TINY 1.0e-20;
/* 

   matrix is always 3 x 3 , solution is for full system for n == 3,
   for upper 2 x 2 only for n = 2

 */
void hc_ludcmp_3x3(HC_PREC a[3][3],int n,int *indx)
{
  int i,imax=0,j,k;
  HC_PREC big,dum,sum,temp;
  HC_PREC vv[3];
  
  for (i=0;i < n;i++) {
    big=0.0;
    for (j=0;j < n;j++)
      if ((temp = fabs(a[i][j])) > big) 
	big=temp;
    if (fabs(big) < HC_EPS_PREC) {
      fprintf(stderr,"hc_ludcmp_3x3: singular matrix in routine, big: %g\n",
	      big);
      for(j=0;j <n;j++){
	for(k=0;k<n;k++)
	  fprintf(stderr,"%g ",a[j][k]);
	fprintf(stderr,"\n");
      }
      exit(-1);
    }
    vv[i]=1.0/big;
  }
  for (j=0;j < n;j++) {
    for (i=0;i < j;i++) {
      sum = a[i][j];
      for (k=0;k < i;k++) 
	sum -= a[i][k] * a[k][j];
      a[i][j]=sum;
    }
    big=0.0;
    for (i=j;i < n;i++) {
      sum=a[i][j];
      for (k=0;k < j;k++)
	sum -= a[i][k] * a[k][j];
      a[i][j]=sum;
      if ( (dum = vv[i]*fabs(sum)) >= big) {
	big=dum;
	imax=i;
      }
    }
    if (j != imax) {
      for (k=0;k < n;k++) {
	dum = a[imax][k];
	a[imax][k]=a[j][k];
	a[j][k]=dum;
      }
      vv[imax]=vv[j];
    }
    indx[j]=imax;
    if (fabs(a[j][j]) < HC_EPS_PREC) 
      a[j][j] = NR_TINY;
    if (j != 2) {
      dum=1.0/(a[j][j]);
      for (i=j+1;i < n;i++) 
	a[i][j] *= dum;
    }
  }
}

#undef NR_TINY
void hc_lubksb_3x3(HC_PREC a[3][3], int n,int *indx, HC_PREC *b)
{
  int i,ii=0,ip,j;
  HC_PREC sum;
  int nm1;
  nm1 = n - 1;
  for (i=0;i < n;i++) {
    ip = indx[i];
    sum = b[ip];
    b[ip]=b[i];
    if (ii)
      for (j=ii-1;j <= i-1;j++) 
	sum -= a[i][j]*b[j];
    else if (fabs(sum)>HC_EPS_PREC) 
      ii = i+1;
    b[i]=sum;
  }
  for (i=nm1;i>=0;i--) {
    sum=b[i];
    for (j=i+1;j < n;j++) 
      sum -= a[i][j]*b[j];
    b[i] = sum/a[i][i];
  }
}

#endif

