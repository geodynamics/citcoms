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
void rtp2xyz(float , float , float, float *);
void convert_pvec_to_cvec(float ,float , float , float *,float *);
void *safe_malloc (size_t );
void myerror(struct All_variables *,char *);
void xyz2rtp(float ,float ,float ,float *);
void get_r_spacing_fine(double *,struct All_variables *);
void get_r_spacing_at_levels(double *,struct All_variables *);


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

  for(m=1; m<=E->sphere.caps_per_proc; m++) {
    E->sbc.node[m] = (int* ) malloc((E->lmesh.nno+1)*sizeof(int));

    n = 1;
    for(i=1; i<=E->lmesh.nno; i++) {
      if(E->node[m][i] & sbc_flags) {
	E->sbc.node[m][i] = n;
	n++;
      }
      else
	E->sbc.node[m][i] = 0;

    }

    for(side=SIDE_BEGIN; side<=SIDE_END; side++)
      for(d=1; d<=E->mesh.nsd; d++) {
	E->sbc.SB[m][side][d] = (double *) malloc(n*sizeof(double));

	for(i=0; i<n; i++)
	  E->sbc.SB[m][side][d][i] = 0;
      }

    for(d=1; d<=E->mesh.nsd; d++)
      for(i=1; i<=E->lmesh.nno; i++)
	if(E->node[m][i] & sbc_flag[d] && E->sphere.cap[m].VB[d][i] != 0) {
	  j = E->sbc.node[m][i];
	  for(side=SIDE_BOTTOM; side<=SIDE_TOP; side++)
	    E->sbc.SB[m][side][d][j] = E->sphere.cap[m].VB[d][i];
	}
  }
}


void get_buoyancy(struct All_variables *E, double **buoy)
{
    int i,j,m;
    double temp,temp2;
    void remove_horiz_ave2(struct All_variables*, double**);

    /* Rayleigh number */
    temp = E->control.Atemp;

    /* thermal buoyancy */
    for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(i=1;i<=E->lmesh.nno;i++) {
        int nz = ((i-1) % E->lmesh.noz) + 1;
        /* We don't need to substract adiabatic T profile from T here,
         * since the horizontal average of buoy will be removed.
         */
        buoy[m][i] =  temp * E->refstate.rho[nz]
            * E->refstate.thermal_expansivity[nz] * E->T[m][i];
    }

    /* chemical buoyancy */
    if(E->control.tracer &&
       (E->composition.ichemical_buoyancy)) {
        for(j=0;j<E->composition.ncomp;j++) {
            /* TODO: how to scale chemical buoyancy wrt reference density? */
            temp2 = E->composition.buoyancy_ratio[j] * temp;
            for(m=1;m<=E->sphere.caps_per_proc;m++)
                for(i=1;i<=E->lmesh.nno;i++)
                    buoy[m][i] -= temp2 * E->composition.comp_node[m][j][i];
        }
    }

    /* phase change buoyancy */
    phase_change_apply_410(E, buoy);
    phase_change_apply_670(E, buoy);
    phase_change_apply_cmb(E, buoy);

     /* convert density to buoyancy */
     for(m=1;m<=E->sphere.caps_per_proc;m++)
       for(i=1;i<=E->lmesh.noz;i++)
             for(j=0;j<E->lmesh.nox*E->lmesh.noy;j++) {
                 int n = j*E->lmesh.noz + i;
                 buoy[m][n] *= E->refstate.gravity[i];
             }

    remove_horiz_ave2(E,buoy);

    return;
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
    /**/
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


/* Read in a file containing previous values of a field. The input in the parameter
   file for this should look like: `previous_name_file=string' and `previous_name_column=int'
   where `name' is substituted by the argument of the function.

   The file should have the standard CITCOM output format:
     # HEADER LINES etc
     index X Z Y ... field_value1 ...
     index X Z Y ... field_value2 ...
   where index is the node number, X Z Y are the coordinates and
   the field value is in the column specified by the abbr term in the function argument

   If the number of nodes OR the XZY coordinates for the node number (to within a small tolerance)
   are not in agreement with the existing mesh, the data is interpolated.

   */

int read_previous_field(E,field,name,abbr)
    struct All_variables *E;
    float **field;
    char *name, *abbr;
{
    int input_string();

    char discard[5001];
    char *token;
    char *filename;
    char *input_token;
    FILE *fp;
    int fnodesx,fnodesz,fnodesy;
    int i,j,column,found,m;

    float *X,*Z,*Y;

    filename=(char *)malloc(500*sizeof(char));
    input_token=(char *)malloc(1000*sizeof(char));

    /* Define field name, read parameter file to determine file name and column number */

    sprintf(input_token,"previous_%s_file",name);
    if(!input_string(input_token,filename,"initialize",E->parallel.me)) {
	fprintf(E->fp,"No previous %s information found in input file\n",name);fflush(E->fp);
	return(0);   /* if not found, take no further action, return zero */
    }


    fprintf(E->fp,"Previous %s information is in file %s\n",name,filename);fflush(E->fp);

    /* Try opening the file, fatal if this fails too */

    if((fp=fopen(filename,"r")) == NULL) {
	fprintf(E->fp,"Unable to open the required file `%s' (this is fatal)",filename);
	fflush(E->fp);

	parallel_process_termination();
    }


     /* Read header, get nodes xzy */

    fgets(discard,4999,fp);
    fgets(discard,4999,fp);
    i=sscanf(discard,"# NODESX=%d NODESZ=%d NODESY=%d",&fnodesx,&fnodesz,&fnodesy);
    if(i<3) {
	fprintf(E->fp,"File %s is not in the correct format\n",filename);fflush(E->fp);
	exit(1);
    }

    fgets(discard,4999,fp); /* largely irrelevant line */
    fgets(discard,4999,fp);

    /* this last line is the column headers, we need to search for the occurence of abbr to
       find out the column to be read in */

    if(strtok(discard,"|")==NULL) {
	fprintf(E->fp,"Unable to deciphre the columns in the input file");fflush(E->fp);
	exit(1);
    }

    found=0;
    column=1;

    while(found==0 && (token=strtok(NULL,"|")) != NULL) {
	if(strstr(token,abbr)!=0)
	    found=1;
	column++;
    }

    if(found) {
	fprintf(E->fp,"\t%s (%s) found in column %d\n",name,abbr,column);fflush(E->fp);
    }
    else {
	fprintf(E->fp,"\t%s (%s) not found in file: %s\n",name,abbr,filename);fflush(E->fp);
	exit(1);
    }



    /* Another fatal condition (not suitable for interpolation: */
    if(((3!= E->mesh.nsd) && (fnodesy !=1)) || ((3==E->mesh.nsd) && (1==fnodesy))) {
	fprintf(E->fp,"Input data for file `%s'  is of inappropriate dimension (not %dD)\n",filename,E->mesh.nsd);fflush(E->fp);
	exit(1);
    }

    if(fnodesx != E->lmesh.nox || fnodesz != E->lmesh.noz || fnodesy != E->lmesh.noy) {
       fprintf(stderr,"wrong dimension in the input temperature file!!!!\n");
       exit(1);
       }

    X=(float *)malloc((2+fnodesx*fnodesz*fnodesy)*sizeof(float));
    Z=(float *)malloc((2+fnodesx*fnodesz*fnodesy)*sizeof(float));
    Y=(float *)malloc((2+fnodesx*fnodesz*fnodesy)*sizeof(float));

   /* Format for reading the input file (including coordinates) */

    sprintf(input_token," %%d %%e %%e %%e");
    for(i=5;i<column;i++)
	strcat(input_token," %*f");
    strcat(input_token," %f");


    for(m=1;m<=E->sphere.caps_per_proc;m++)
      for(i=1;i<=fnodesx*fnodesz*fnodesy;i++) {
	fgets(discard,4999,fp);
	sscanf(discard,input_token,&j,&(X[i]),&(Z[i]),&(Y[i]),&field[m][i]);
        }
    /* check consistency & need for interpolation */

    fclose(fp);


    free((void *)X);
    free((void *)Z);
    free((void *)Y);
    free((void *)filename);
    free((void *)input_token);

    return(1);
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


/* compute base vectors for conversion of polar to cartesian vectors
   base[9]
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
  if(fabs(E->control.rrlayer[0] -E->sphere.ri) > 1e-6)
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
