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
    system(command);
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

    temp = E->control.Atemp;

    for(m=1;m<=E->sphere.caps_per_proc;m++)
	for(i=1;i<=E->lmesh.nno;i++) {
	    int nz = ((i-1) % E->lmesh.noz) + 1;
            /* We don't need to substract adiabatic T profile from T here,
             * since the horizontal average of buoy will be removed.
             */
	    buoy[m][i] =  temp * E->refstate.rho[nz]
                * E->refstate.expansivity[nz] * E->T[m][i];
	}

    /* chemical buoyancy */
    if(E->control.tracer && E->composition.ichemical_buoyancy==1) {
        for(j=0;j<E->composition.ncomp;j++) {
            temp2 = E->composition.buoyancy_ratio[j] * temp;
            for(m=1;m<=E->sphere.caps_per_proc;m++)
                for(i=1;i<=E->lmesh.nno;i++)
                    buoy[m][i] -= temp2 * E->composition.comp_node[m][j][i];
        }
    }

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

    remove_horiz_ave2(E, buoy);
    return;
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
