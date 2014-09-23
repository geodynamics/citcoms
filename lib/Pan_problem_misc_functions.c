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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "element_definitions.h"
#include "global_defs.h"

#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#if defined(__sgi) || defined(__osf__)
#include <sys/types.h>
#endif

#include "phase_change.h"
#include "parallel_related.h"

void calc_cbase_at_tp(float , float , float *);
void calc_cbase_at_tp_d(double , double , double *);
void rtp2xyz(float , float , float, float *);
void rtp2xyzd(double , double , double, double *);
void convert_pvec_to_cvec(float ,float , float , float *,float *);
void convert_pvec_to_cvec_d(double ,double , double , double *,double *);
void *safe_malloc (size_t );
void myerror(struct All_variables *,char *);
void xyz2rtp(float ,float ,float ,float *);
void xyz2rtpd(float ,float ,float ,double *);
void get_r_spacing_fine(double *,struct All_variables *);
void get_r_spacing_at_levels(double *,struct All_variables *);
void calc_cbase_at_node(int , float *,struct All_variables *);
#ifdef ALLOW_ELLIPTICAL
double theta_g(double , struct All_variables *);
#endif
#ifdef USE_GGRD
void ggrd_adjust_tbl_rayleigh(struct All_variables *,double **);
#endif

int get_process_identifier()
{
    int pid;

    pid = (int) getpid();
    return(pid);
}


void unique_copy_file(E,name,comment)
    struct All_variables *E;
    char *name, *comment;
{
    char unique_name[500];
    char command[600];

   if (E->parallel.me==0) {
    sprintf(unique_name,"%06d.%s-%s",E->control.PID,comment,name);
    sprintf(command,"cp -f %s %s\n",name,unique_name);
#if 0
    /* disable copying file, since some MPI implementation doesn't support it */
    system(command);
#endif
    }

}


void apply_side_sbc(struct All_variables *E)
{
  /* This function is called only when E->control.side_sbcs is true.
     Purpose: convert the original b.c. data structure, which only supports
              SBC on top/bottom surfaces, to new data structure, which supports
	      SBC on all (6) sides
  */
  int i, j, d, m, side, n;
  const unsigned sbc_flags = SBX | SBY | SBZ;
  const unsigned sbc_flag[4] = {0,SBX,SBY,SBZ};

  if(E->parallel.total_surf_proc==12) {
    fprintf(stderr, "side_sbc is applicable only in Regional version\n");
    parallel_process_termination();
  }

    E->sbc.node[CPPR] = (int* ) malloc((E->lmesh.nno+1)*sizeof(int));

    n = 1;
    for(i=1; i<=E->lmesh.nno; i++) {
      if(E->node[i] & sbc_flags) {
	E->sbc.node[CPPR][i] = n;
	n++;
      }
      else
	E->sbc.node[CPPR][i] = 0;

    }

    for(side=SIDE_BEGIN; side<=SIDE_END; side++)
      for(d=1; d<=E->mesh.nsd; d++) {
	E->sbc.SB[CPPR][side][d] = (double *) malloc(n*sizeof(double));

	for(i=0; i<n; i++)
	  E->sbc.SB[CPPR][side][d][i] = 0;
      }

    for(d=1; d<=E->mesh.nsd; d++)
      for(i=1; i<=E->lmesh.nno; i++)
	if(E->node[i] & sbc_flag[d] && E->sphere.cap[CPPR].VB[d][i] != 0) {
	  j = E->sbc.node[CPPR][i];
	  for(side=SIDE_BOTTOM; side<=SIDE_TOP; side++)
	    E->sbc.SB[CPPR][side][d][j] = E->sphere.cap[CPPR].VB[d][i];
	}
}


void get_buoyancy(struct All_variables *E, double **buoy)
{
    int i,j,m,n,nz,nxny;
    int lev = E->mesh.levmax;
    double temp,temp2,rfac,cost2;
    void remove_horiz_ave2(struct All_variables*, double**);
    //char filename[100];FILE *out;

    nxny = E->lmesh.nox*E->lmesh.noy;
    /* Rayleigh number (can be negative for time reversal) */
    temp = E->control.Atemp;

    /* thermal buoyancy */
      for(i=1;i<=E->lmesh.nno;i++) {
	nz = ((i-1) % E->lmesh.noz) + 1;
        /* We don't need to substract adiabatic T profile from T here,
         * since the horizontal average of buoy will be removed.
         */
        buoy[CPPR][i] =  temp * E->refstate.rho[nz]
	  * E->refstate.thermal_expansivity[nz] * E->T[i];
      }
    
    /* chemical buoyancy */
    if(E->control.tracer &&
       (E->composition.ichemical_buoyancy)) {
      for(j=0;j<E->composition.ncomp;j++) {
	/* TODO: how to scale chemical buoyancy wrt reference density? */
	temp2 = E->composition.buoyancy_ratio[j] * temp;
	      for(i=1;i<=E->lmesh.nno;i++)
		buoy[CPPR][i] -= temp2 * E->composition.comp_node[j][i];
      }
    }
#ifdef USE_GGRD
    /* surface layer Rayleigh modification? */
    if(E->control.ggrd.ray_control)
      ggrd_adjust_tbl_rayleigh(E,buoy);
#endif
    /* phase change buoyancy */
    phase_change_apply_410(E, buoy);
    phase_change_apply_670(E, buoy);
    phase_change_apply_cmb(E, buoy);

    /* 
       convert density to buoyancy 
    */
#ifdef ALLOW_ELLIPTICAL
    if(E->data.use_rotation_g){
      /* 

      rotational correction, the if should not add significant
      computational burden

      */
      /* g= g_e (1+(5/2m-f) cos^2(theta)) , not theta_g */
      rfac = E->data.ge*(5./2.*E->data.rotm-E->data.ellipticity);
      /*  */
	for(j=0;j < nxny;j++) {
	  for(i=1;i<=E->lmesh.noz;i++)
	    n = j*E->lmesh.noz + i; /* this could be improved by only
				       computing the cos as a function
				       of lat, but leave for now  */
	    cost2 = cos(E->sx[CPPR][1][n]);cost2 = cost2*cost2;	    /* cos^2(theta) */
	    /* correct gravity for rotation */
	    buoy[CPPR][n] *= E->refstate.gravity[i] * (E->data.ge+rfac*cost2);
	  }
    }else{
#endif
      /* default */
      /* no latitude dependency of gravity */
	for(j=0;j < nxny;j++) {
	  for(i=1;i<=E->lmesh.noz;i++){
	    n = j*E->lmesh.noz + i;
	    buoy[CPPR][n] *= E->refstate.gravity[i];
	  }
	}
#ifdef ALLOW_ELLIPTICAL
    }
#endif    
    

    remove_horiz_ave2(E,buoy);
    
}


/*
 * Scan input str to get a double vector *values. The vector length is from
 * input len. The input str contains white-space seperated numbers. Return
 * the number of columns read (can be less than len).
 */
static int scan_double_vector(const char *str, int len, double *values)
{
    char *nptr, *endptr;
    int i;

    /* cast to avoid compiler warning */
    nptr = endptr = (char *) str;

    for (i = 0; i < len; ++i) {
        values[i] = strtod(nptr, &endptr);
        if (nptr == endptr) {
            /* error: no conversion is performed */
            return i;
        }
        nptr = endptr;
    }

    /** debug **
    for (i = 0; i < len; ++i) fprintf(stderr, "%e, ", values[i]);
    fprintf(stderr, "\n");
    */
    return len;
}


/*
 * From input file, read a line, which contains white-space seperated numbers
 * of lenght num_columns, store the numbers in a double array, return the
 * number of columns read (can be less than num_columns).
 */
int read_double_vector(FILE *in, int num_columns, double *fields)
{
    char buffer[256], *p;

    p = fgets(buffer, 255, in);
    if (!p) {
        return 0;
    }

    return scan_double_vector(buffer, num_columns, fields);
}


/*
 * Read fp line by line, until a line matching string param is found.
 * Then, the next E->mesh.nel lines are read into var array.
 */
void read_visc_param_from_file(struct All_variables *E,
                               const char *param, float *var,
                               FILE *fp)
{
    char buffer[256], *p;
    size_t len;
    int i;

    len = strlen(param);

    /* back to beginning of file */
    rewind(fp);

    while(1) {
        p = fgets(buffer, 255, fp);
        if(!p) {
            /* reaching end of file */
            if(E->parallel.me == 0)
                fprintf(stderr, "Cannot find param '%s' in visc_layer_file\n", param);
            parallel_process_termination();
        }

        if(strncmp(buffer, param, len) == 0)
            /* find matching param */
            break;
    }

    /* fill in the array in reversed order */
    for(i=E->mesh.elz-1; i>=0; i--) {
        int n;
        n = fscanf(fp, "%f", &(var[i]));
        //fprintf(stderr, "%d %f\n", i, var[i]);
        if(n != 1) {
            fprintf(stderr,"Error while reading file '%s'\n", E->viscosity.layer_file);
            exit(8);
        }
    }

}



/* =================================================
  my version of arc tan
 =================================================*/

double myatan(y,x)
 double y,x;
 {
 double fi;

 fi = atan2(y,x);

 if (fi<0.0)
    fi += 2*M_PI;

 return(fi);
 }


double return1_test()
{
 return 1.0;
}

/* convert r,theta,phi system to cartesian, xout[3]
   there's a double version of this in Tracer_setup called
   sphere_to_cart

*/
void rtp2xyzd(double r, double theta, double phi, double *xout)
{
  double rst;
  rst = r * sin(theta);
  xout[0] = rst * cos(phi);	/* x */
  xout[1] = rst * sin(phi); 	/* y */
  xout[2] = r * cos(theta);
}
/* float version */
void rtp2xyz(float r, float theta, float phi, float *xout)
{
  float rst;
  rst = r * sin(theta);
  xout[0] = rst * cos(phi);	/* x */
  xout[1] = rst * sin(phi); 	/* y */
  xout[2] = r * cos(theta);
}
void xyz2rtp(float x,float y,float z,float *rout)
{
  float tmp1,tmp2;
  tmp1 = x*x + y*y;
  tmp2 = tmp1 + z*z;
  rout[0] = sqrt(tmp2);		/* r */
  rout[1] = atan2(sqrt(tmp1),z); /* theta */
  rout[2] = atan2(y,x);		/* phi */
}
void xyz2rtpd(float x,float y,float z,double *rout)
{
  double tmp1,tmp2;
  tmp1 = (double)x*(double)x + (double)y*(double)y;
  tmp2 = tmp1 + (double)z*(double)z;
  rout[0] = sqrt(tmp2);		/* r */
  rout[1] = atan2(sqrt(tmp1),(double)z); /* theta */
  rout[2] = atan2((double)y,(double)x);		/* phi */
}


/* compute base vectors for conversion of polar to cartesian vectors
   base[9], i.e. those are the cartesian representation of the r,
   theta, and phi basis vectors at theta, phi
*/
void calc_cbase_at_tp(float theta, float phi, float *base)
{


 double ct,cp,st,sp;

 ct=cos(theta);
 cp=cos(phi);
 st=sin(theta);
 sp=sin(phi);
 /* r */
 base[0]= st * cp;
 base[1]= st * sp;
 base[2]= ct;
 /* theta */
 base[3]= ct * cp;
 base[4]= ct * sp;
 base[5]= -st;
 /* phi */
 base[6]= -sp;
 base[7]= cp;
 base[8]= 0.0;
}
void calc_cbase_at_tp_d(double theta, double phi, double *base) /* double version */
{


 double ct,cp,st,sp;

 ct=cos(theta);
 cp=cos(phi);
 st=sin(theta);
 sp=sin(phi);
 /* r */
 base[0]= st * cp;
 base[1]= st * sp;
 base[2]= ct;
 /* theta */
 base[3]= ct * cp;
 base[4]= ct * sp;
 base[5]= -st;
 /* phi */
 base[6]= -sp;
 base[7]= cp;
 base[8]= 0.0;
}

/* calculate base at nodal locations where we have precomputed cos/sin */

void calc_cbase_at_node(int node, float *base,struct All_variables *E)
{
  int lev ;
  double ct,cp,st,sp;
  lev = E->mesh.levmax;
  st = E->SinCos[lev][CPPR][0][node]; /* for elliptical, sincos would be  corrected */
  sp = E->SinCos[lev][CPPR][1][node];
  ct = E->SinCos[lev][CPPR][2][node];
  cp = E->SinCos[lev][CPPR][3][node];
           
  /* r */
  base[0]= st * cp;
  base[1]= st * sp;
  base[2]= ct;
  /* theta */
  base[3]= ct * cp;
  base[4]= ct * sp;
  base[5]= -st;
  /* phi */
  base[6]= -sp;
  base[7]= cp;
  base[8]= 0.0;
}

/* given a base from calc_cbase_at_tp, convert a polar vector to
   cartesian */
void convert_pvec_to_cvec(float vr,float vt,
			  float vp, float *base,
			  float *cvec)
{
  int i;
  for(i=0;i<3;i++){
    cvec[i]  = base[i]  * vr;
    cvec[i] += base[3+i]* vt;
    cvec[i] += base[6+i]* vp;
  }
}
void convert_pvec_to_cvec_d(double vr,double vt,
			    double vp, double *base,
			    double *cvec)
{
  int i;
  for(i=0;i<3;i++){
    cvec[i]  = base[i]  * vr;
    cvec[i] += base[3+i]* vt;
    cvec[i] += base[6+i]* vp;
  }
}
/*
   like malloc, but with test

   similar to Malloc1 but I didn't like the int as argument

*/
void *safe_malloc (size_t size)
{
  void *tmp;

  if ((tmp = malloc(size)) == NULL) {
    fprintf(stderr, "safe_malloc: could not allocate memory, %.3f MB\n",
	    (float)size/(1024*1024.));
    parallel_process_termination();
  }
  return (tmp);
}
/* error handling routine, TWB */

void myerror(struct All_variables *E,char *message)
{
  void record();

  E->control.verbose = 1;
  record(E,message);
  fprintf(stderr,"node %3i: error: %s\n",
	  E->parallel.me,message);
  parallel_process_termination();
}



/*



attempt to space rr[1...nz] such that bfrac*nz nodes will be within the lower
brange fraction of (ro-ri), and similar for the top layer

function below is more general

*/
void get_r_spacing_fine(double *rr, struct All_variables *E)
{
  int k,klim,nb,nt,nm;
  double drb,dr0,drt,dr,drm,range,r,mrange, brange,bfrac,trange, tfrac;

  brange = (double)E->control.coor_refine[0];
  bfrac =  (double)E->control.coor_refine[1];
  trange = (double)E->control.coor_refine[2];
  tfrac = (double)E->control.coor_refine[3];

  range = (double) E->sphere.ro - E->sphere.ri;		/* original range */

  mrange = 1 - brange - trange;
  if(mrange <= 0)
    myerror(E,"get_r_spacing_fine: bottom and top range too large");

  brange *= range;		/* bottom */
  trange *= range;		/* top */
  mrange *= range;		/* middle */

  nb = E->mesh.noz * bfrac;
  nt = E->mesh.noz * tfrac;
  nm = E->mesh.noz - nb - nt;
  if((nm < 1)||(nt < 2)||(nb < 2))
    myerror(E,"get_r_spacing_fine: refinement out of bounds");

  drb = brange/(nb-1);
  drt = trange/(nt-1);
  drm = mrange / (nm  + 1);

  for(r=E->sphere.ri,k=1;k<=nb;k++,r+=drb){
    rr[k] = r;
  }
  klim = E->mesh.noz - nt + 1;
  for(r=r-drb+drm;k < klim;k++,r+=drm){
    rr[k] = r;
  }
  for(;k <= E->mesh.noz;k++,r+=drt){
    rr[k] = r;
  }
}
/*


get r spacing at radial locations and node numbers as specified
CitcomCU style

rr[1...E->mesh.noz]


e.g.:

	r_grid_layers=3		# minus 1 is number of layers with uniform grid in r
	rr=0.5,0.75,1.0 	#    starting and ending r coodinates
	nr=1,37,97		#    starting and ending node in r direction

*/
void get_r_spacing_at_levels(double *rr,struct All_variables *E)
{
  double ddr;
  int k,j;
  /* do some sanity checks */
  if(E->control.nrlayer[0] != 1)
    myerror(E,"first node for coor=3 should be unity");
  if(E->control.nrlayer[E->control.rlayers-1] != E->mesh.noz)
      myerror(E,"last node for coor = 3 input should match max nr z nodes");
  if(fabs(E->control.rrlayer[0] -E->sphere.ri) > 1e-5)
    myerror(E,"inner layer for coor=3 input should be inner radius");
  if(fabs(E->control.rrlayer[ E->control.rlayers-1] - E->sphere.ro)>1e-6)
    myerror(E,"outer layer for coor=3 input should be inner radius");
  if(E->control.rlayers < 2)
    myerror(E,"number of rlayers needs to be at leats two for coor = 3");

  rr[1] =  E->control.rrlayer[0];
  for(j = 1; j < E->control.rlayers; j++){
    ddr = (E->control.rrlayer[j] - E->control.rrlayer[j - 1]) /
      (E->control.nrlayer[j] - E->control.nrlayer[j - 1]);
    for(k = E->control.nrlayer[j-1]+1;k <= E->control.nrlayer[j];k++)
      rr[k] = rr[k-1]+ddr;
    }

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
/* 
   C = A * B

   for 3x3 matrix
   
*/
void matmul_3x3(double a[3][3],double b[3][3],double c[3][3])
{
  int i,j,k;
  double tmp;
  for(i=0;i < 3;i++)
    for(j=0;j < 3;j++){
      tmp = 0.;
      for(k=0;k < 3;k++)
	tmp += a[i][k] * b[k][j];
      c[i][j] = tmp;
    }
}
void remove_trace_3x3(double a[3][3])
{
  double trace;
  trace = (a[0][0]+a[1][1]+a[2][2])/3;
  a[0][0] -= trace;
  a[1][1] -= trace;
  a[2][2] -= trace;
}
void get_9vec_from_3x3(double *l,double vgm[3][3])
{
  l[0] = vgm[0][0];l[1] = vgm[0][1];l[2] = vgm[0][2];
  l[3] = vgm[1][0];l[4] = vgm[1][1];l[5] = vgm[1][2];
  l[6] = vgm[2][0];l[7] = vgm[2][1];l[8] = vgm[2][2];
}
void get_3x3_from_9vec(double l[3][3], double *l9)
{
  l[0][0]=l9[0];  l[0][1]=l9[1];  l[0][2]=l9[2];
  l[1][0]=l9[3];  l[1][1]=l9[4];  l[1][2]=l9[5];
  l[2][0]=l9[6];  l[2][1]=l9[7];  l[2][2]=l9[8];
}

#ifdef ALLOW_ELLIPTICAL
/* correct from spherical coordinate system theta to an ellipsoidal
   theta_g which corresponds to the local base vector direction in
   theta */
double theta_g(double theta, struct All_variables *E)
{
  double tmp;

  if(E->data.use_ellipse){
    tmp = M_PI_2 - theta;
    return M_PI_2 - atan2(tan(tmp),E->data.efac);
  }else{
    return theta;
  }
}
#endif
